import os
import json
import asyncio
import pandas as pd
from PIL import Image
import io
import glob
import re
import time
import datetime
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

IMAGE_DIR = ""
CSV_PATH = ""
TARGET_SIZE = (1092, 1092)
MAX_CONCURRENT = 20
OUTPUT_DIR = ""
API_KEY = ""
MODEL_ID = "gemini-2.5-pro-preview-03-25"

os.makedirs(OUTPUT_DIR, exist_ok=True)

log_file = os.path.join(OUTPUT_DIR, "processing_log.txt")
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"Processing started - {datetime.datetime.now()}\n")

async def log_message(message):
    print(message)
    async with asyncio.Lock():
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now()} - {message}\n")

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

client = genai.Client(api_key=API_KEY)

async def load_csv():
    try:
        df = pd.read_csv(CSV_PATH)
        await log_message(f"Successfully loaded CSV file with {len(df)} rows")
        await log_message(f"CSV columns: {df.columns.tolist()}")
        
        id_column = next((col for col in df.columns if 'new_id' in col.lower()), 
                     next((col for col in df.columns if 'id' in col.lower()), None))
        
        lat_column = next((col for col in df.columns if 'lat' in col.lower()), None)
        lon_column = next((col for col in df.columns if 'lon' in col.lower()), None)
        region_column = next((col for col in df.columns if 'region' in col.lower()), None)
        
        if id_column:
            await log_message(f"Using '{id_column}' as ID column")
            await log_message(f"Sample IDs: {df[id_column].head().tolist()}")
        else:
            await log_message("Warning: Cannot identify ID column")
            
        if lat_column and lon_column:
            await log_message(f"Using '{lat_column}' and '{lon_column}' as coordinate columns")
        else:
            await log_message("Warning: Cannot identify coordinate columns")
            
        return df, id_column, lat_column, lon_column, region_column
    except Exception as e:
        await log_message(f"Error reading CSV file: {e}")
        df = pd.DataFrame(columns=['id', 'lat', 'lon', 'region'])
        return df, 'id', 'lat', 'lon', 'region'

def clean_json_string(json_text):
    cleaned_text = json_text.strip()
    
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def is_valid_json(response_text):
    cleaned_text = clean_json_string(response_text)
    
    try:
        json_data = json.loads(cleaned_text)
        return True, json_data
    except json.JSONDecodeError:
        return False, None

geo_analysis_prompt = """
# Expert Geographical Location Analysis Task

## Overview
I will provide you with an image of a location along with its actual coordinates (latitude and longitude). You are tasked with performing an extremely detailed geographical analysis, identifying micro-features that would help determine the precise location. This requires meticulous examination of all visual elements at both macro and micro scales.

CRITICAL RESPONSE FORMAT INSTRUCTIONS: \
1. Your ENTIRE response must be ONLY a valid JSON object. \
2. DO NOT include ```json markers or any other markdown. \
3. DO NOT add any explanatory text before or after the JSON. \
4. Your response must be a valid JSON object starting with { and ending with }. All keys must be in double quotes. \
5. If you are writing content within JSON format, you need to use escape characters like \{ or \} \
6. Verify your JSON is valid before submitting." 

# Expert Geographical Location Analysis Task

## Required JSON Output Format
Your analysis must be provided in a structured JSON format as follows:
```json
{
  "image_id": "filename.jpg",
"thinking": {
  "25km_analysis": {
    "terrain_characteristics": "Detailed description of regional terrain",
    "climate_features": "Climate zone, seasonal indicators, weather patterns",
    "regional_development": "Historical context, development patterns, land use"
  },
  "1km_analysis": {
    "road_engineering": {
      "surface_materials": "Precise description of road material composition, texture, color, wear patterns, age indicators",
      "line_markings": "Width, color, pattern, material, conformity to regional standards",
      "guardrails": "Design type, material, height, installation pattern, regional specificity",
      "lighting": "Pole design, height, spacing, lamp type, regional standards",
      "drainage": "Design, materials, integration with landscape, climatic adaptation"
    },
    "vegetation": {
      "tree_species": "Scientific names, distribution patterns, growth characteristics, management style",
      "herbaceous_plants": "Undergrowth composition, seasonal state, native vs. introduced species",
      "vegetation_management": "Pruning techniques, planting patterns, maintenance regimes"
    }
  },
  "buildings_infrastructure": {
    "architectural_styles": "Building design elements, materials, roof styles, regional influences",
    "municipal_facilities": "Public infrastructure, signage systems, urban furniture design",
    "transportation_facilities": "Transit design, stops/stations, pedestrian accommodations"
  },
  "vehicles_traffic": {
    "vehicle_types": "Common models, regional preferences, vehicle adaptations",
    "traffic_patterns": "Flow characteristics, driver behaviors, local conventions"
  },
  "terrain_soil": {
    "terrain_details": "Micro-topography, slope characteristics, erosion patterns",
    "soil_characteristics": "Color, texture, composition, regional distinctiveness"
  },
  "elimination_similar_regions": [
    {
      "eliminated_region": "Region name",
      "reasons": [
        "Specific distinguishing feature 1 that excludes this region",
        "Specific distinguishing feature 2 that excludes this region",
        "Specific distinguishing feature 3 that excludes this region"
      ]
    }
  ],
  }
    "answer": {
    "coordinates": {
    "latitude": 00.000000,
    "longitude": 00.000000,
    "region": "Region name"
  },
  "final_location_confirmation": {
    "confirmed_location": "Precise description of location",
    "decisive_features": [
      "Highly distinctive feature 1 unique to this location",
      "Highly distinctive feature 2 unique to this location",
      "Highly distinctive feature 3 unique to this location"
    ],
    "accuracy_assessment": "Confidence level with justification"
  }
  }
}
Comprehensive Analysis Requirements

25km Radius Analysis
Examine the broader geographical context:

Terrain characteristics: Identify landforms, elevation patterns, geological formations, and natural features within 25km radius
Climate features: Determine climate zone, vegetation patterns indicating precipitation and temperature regimes
Regional development: Assess human settlement patterns, infrastructure networks, land use systems, and regional planning approaches

1km Radius Analysis
Perform micro-level analysis with extraordinary attention to detail:
Road Engineering System Micro-Analysis

Surface materials analysis:

Identify exact pavement type (SMA asphalt, concrete, chip seal, etc.)
Measure visible aggregate size, texture depth, and wear patterns
Note surface color variations, patching techniques, and aging indicators
Assess maintenance standards and conformity to regional practices


Line marking system:

Measure precise width of lines (in cm)
Determine material type (thermoplastic, paint, preformed tape)
Analyze pattern dimensions (line length, gap spacing, edge definition)
Identify conformity to national or regional standards
Note wear level and repainting frequency indicators


Guardrail and safety features:

Document design type, dimensions, and material composition
Measure height, post spacing, and connection methods
Note surface treatments, weathering patterns, and repair approaches
Identify manufacturer-specific design elements and regional standards
Analyze foundation systems and installation techniques


Lighting and electrical infrastructure:

Catalog pole design, height, material, and spacing intervals
Identify luminaire types, mounting configurations, and light distribution patterns
Note electrical supply systems, control units, and auxiliary equipment
Assess maintenance standards and regional design preferences
Determine age of installation through design characteristics


Road foundation and drainage systems:

Analyze cross-section design, crown height, and superelevation angles
Document drainage solutions, curb designs, and water management systems
Assess shouldering techniques, slope treatments, and erosion control methods
Note frost protection measures in applicable regions
Identify regional design adaptations for local climate conditions



Vegetation and Ecological Environment Analysis

Tree species precise identification:

Provide scientific names (genus and species) of visible tree species
Analyze trunk characteristics, bark patterns, and branching structure
Estimate tree age, height, and diameter with precision
Note planting patterns, spacing, and alignment methodology
Identify regional versus imported species and their significance
Document pruning and maintenance techniques characteristic of the region


Herbaceous plants and shrub layer:

Identify ground cover species composition and diversity
Note seasonal growth state, flowering stage, or dormancy indicators
Assess management regime (wild, semi-managed, or cultivated)
Document understory complexity and succession stage
Analyze ecological relationships between vegetation layers


Vegetation management characteristics:

Identify maintenance techniques (pruning styles, trimming heights)
Assess management frequency and professional standards
Note regional vegetation control practices and aesthetic preferences
Document evidence of irrigation, fertilization, or other interventions
Analyze relationship between vegetation management and infrastructure



Building and Infrastructure Micro-Features

Architectural styles and materials:

Identify building designs, proportions, and cultural influences
Analyze facade treatments, window configurations, and roofing systems
Document construction materials and their regional significance
Note decorative elements, color schemes, and aesthetic preferences
Assess building age, renovations, and modifications


Municipal facilities and utilities:

Document public infrastructure designs and standards
Analyze utility distribution systems and access points
Note street furniture design, materials, and placement patterns
Identify waste management systems and environmental controls
Assess infrastructure maintenance standards and quality


Transportation infrastructure specifics:

Analyze transit stop designs, materials, and information systems
Document pedestrian facilities, crossing designs, and accessibility features
Note traffic control devices, signalization systems, and management approaches
Identify transportation network hierarchies and connectivity patterns



Vehicle and Traffic Pattern Analysis

Vehicle types and regional indicators:

Identify specific vehicle makes, models, and production years
Note predominant vehicle classes and their regional significance
Document vehicle modifications or adaptations to local conditions
Analyze fleet age, condition, and maintenance standards
Assess vehicle registration, identification, and regulatory indicators


Traffic behavior patterns:

Analyze driving conventions, spacing patterns, and lane discipline
Note parking practices, stopping behaviors, and driver interactions
Document traffic volumes, peak patterns, and congestion indicators
Identify unique local traffic management solutions



Terrain and Soil Micro-Analysis

Terrain micro-features:

Document slope gradients, aspect orientations, and surface undulations
Analyze water flow patterns, erosion features, and deposition areas
Note terrain modifications, grading practices, and land forming techniques
Identify geomorphological processes evident at micro-scale


Soil characteristics:

Determine soil color using Munsell notation or precise descriptors
Analyze visible texture, structure, and compositional elements
Note moisture content, organic matter presence, and horizon development
Identify unique regional soil properties and their implications
Document human alterations to natural soil profiles



Elimination of Similar Regions
For each visually similar region that might be confused with the actual location:

Identify at least 3-5 specific micro-features that definitively exclude this region
Explain why these features could not exist in the misidentified region
Provide concrete, measurable differences rather than generalities
Reference regional standards, regulations, or practices that differ
Compare specific design elements that have high geographical discrimination value

Final Location Confirmation
Based on comprehensive analysis:

Provide precise determination of location with confidence assessment
Highlight the 3-5 most distinctive micro-features that uniquely identify this location
Explain why this specific combination of features creates a geographical "fingerprint"
Address any potential discrepancies or unexpected features
Assess reliability of conclusion based on quantity and quality of diagnostic features

Analysis Methodology Requirements
Your analysis must demonstrate:

Precise measurement rather than estimation: Provide exact dimensions where possible
Scientific taxonomy: Use correct scientific names for biological entities
Technical specificity: Apply proper engineering and architectural terminology
Regional standard references: Cite relevant building codes, design standards, or regional practices
Hierarchical observation: Progress from obvious features to subtle micro-details
Causal relationship analysis: Explain why features exist rather than merely identifying them
Comparative discrimination: Highlight what makes features unique to this location versus similar regions

In your extended thinking process, please:

First, examine the image extremely thoroughly, identifying EVERY possible micro-detail that could help with geographic positioning:

Road materials, markings, and construction techniques
Vegetation species, growth patterns, and management approaches
Architectural styles, building materials, and construction methods
Vehicle types, models, and traffic patterns
Soil types, colors, and terrain characteristics
Any visible signage, text, or cultural indicators
Weather conditions and seasonal indicators


For each identified feature, research and determine:

Which regions in the world commonly have this feature
How distinctive or common this feature is globally
What combination of features creates a unique "geographic fingerprint"


Then, using the provided coordinates, verify your observations:

Research this specific region's characteristics
Determine which of your observations match known features of this region
Identify any discrepancies between your observations and known regional features
Assess your confidence in the location match


Then, create a comprehensive profile of the location that:

Highlights the most distinctive features that confirm its location
Explains why this couldn't be other visually similar locations
Provides extremely specific details that are unique to this region

Finally, when submitting your assignment, ***be sure*** to verify your work to make sure you have not missed any details that could be described, or if any details do not match your description. 

Your final output must be presented as properly formatted JSON according to the structure provided above. Ensure each section contains extremely detailed, specific information derived from meticulous examination of the image.

Remember to be extremely specific and detailed. Use precise measurements, exact species names, specific architectural terms, and reference regional building codes or standards where possible.
"""

def get_geolocation_data(df, image_filename, id_column, lat_column, lon_column, region_column):
    base_filename = os.path.splitext(os.path.basename(image_filename))[0]
    
    try:
        image_data = df[df[id_column] == int(base_filename)]
        
        if image_data.empty:
            image_data = df[df[id_column] == os.path.basename(image_filename)]
            
        if image_data.empty:
            return 0.0, 0.0, "Unknown"
            
        latitude = image_data[lat_column].values[0] if lat_column else 0.0
        longitude = image_data[lon_column].values[0] if lon_column else 0.0
        region = image_data[region_column].values[0] if region_column and not pd.isna(image_data[region_column].values[0]) else "Unknown"
        
        return latitude, longitude, region
    except Exception as e:
        return 0.0, 0.0, "Unknown"

async def process_image_analysis(df, image_path, id_column, lat_column, lon_column, region_column):
    async with semaphore:
        start_time = time.time()
        image_filename = os.path.basename(image_path)
        base_filename = os.path.splitext(image_filename)[0]
        output_base_path = os.path.join(OUTPUT_DIR, base_filename)
        
        if os.path.exists(f"{output_base_path}.json") or os.path.exists(f"{output_base_path}.txt"):
            await log_message(f"[{image_filename}] Already processed - skipping")
            return image_filename, {"status": "skipped", "reason": "already processed"}
        
        latitude, longitude, region = get_geolocation_data(df, image_filename, id_column, lat_column, lon_column, region_column)
        await log_message(f"[{image_filename}] Starting processing: lat={latitude}, lon={longitude}, region={region}")
        
        try:
            image = Image.open(image_path)
            image = image.resize(TARGET_SIZE, Image.LANCZOS)
            
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG", quality=90)
                image_bytes = buffer.getvalue()
                
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            await log_message(f"[{image_filename}] Image processing completed")
        except Exception as e:
            await log_message(f"[{image_filename}] Image processing failed: {e}")
            return image_filename, {"status": "error", "reason": f"image processing failed: {str(e)}"}
        
        custom_prompt = f"{geo_analysis_prompt}\n\nYou are analyzing image '{image_filename}' with coordinates: Latitude: {latitude}, Longitude: {longitude}, Region: {region}.\n\nThoroughly examine all details and provide your complete analysis in the required JSON format."
        
        try:
            google_search_tool = Tool(
                google_search=GoogleSearch()
            )
            
            await log_message(f"[{image_filename}] Starting API call...")
            
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[custom_prompt, image_part],
                config=GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                    temperature=1.0,
                    max_output_tokens=30000,
                )
            )
            
            response_text = response.text
            await log_message(f"[{image_filename}] API response length: {len(response_text)}")
            
            if "```" in response_text:
                await log_message(f"[{image_filename}] Detected markdown markers, cleaning...")
                cleaned_text = clean_json_string(response_text)
                response_text = cleaned_text
            
            is_json, json_data = is_valid_json(response_text)
            
            if is_json:
                json_output_path = f"{output_base_path}.json"
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                await log_message(f"[{image_filename}] Saved as JSON: {json_output_path}")
                save_status = "saved_as_json"
                output_path = json_output_path
            else:
                txt_output_path = f"{output_base_path}.txt"
                with open(txt_output_path, 'w', encoding='utf-8') as f:
                    f.write(response_text)
                await log_message(f"[{image_filename}] Saved as TXT: {txt_output_path}")
                save_status = "saved_as_txt"
                output_path = txt_output_path
            
            end_time = time.time()
            processing_time = end_time - start_time
            await log_message(f"[{image_filename}] Processing completed, time: {processing_time:.2f}s")
            
            return image_filename, {
                "status": "success", 
                "save_format": save_status, 
                "is_valid_json": is_json, 
                "saved_to": output_path, 
                "response_length": len(response_text),
                "processing_time": processing_time
            }
            
        except Exception as e:
            await log_message(f"[{image_filename}] Processing error: {e}")
            import traceback
            trace = traceback.format_exc()
            
            error_result = {"status": "error", "reason": str(e)}
            
            error_output_path = f"{output_base_path}.error.txt"
            with open(error_output_path, 'w', encoding='utf-8') as f:
                f.write(f"Error: {str(e)}\n\nStack trace:\n{trace}")
                
            return image_filename, error_result

async def main():
    df, id_column, lat_column, lon_column, region_column = await load_csv()
    
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    
    def extract_number(filename):
        match = re.search(r'(\d+)', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return 0
    
    image_files.sort(key=extract_number)
    
    await log_message(f"Found {len(image_files)} images to process")
    if image_files:
        await log_message(f"Starting: {os.path.basename(image_files[0])}")
        await log_message(f"Ending: {os.path.basename(image_files[-1])}")
    
    await log_message(f"Creating all processing tasks...")
    tasks = [process_image_analysis(df, image_path, id_column, lat_column, lon_column, region_column) 
             for image_path in image_files]
    
    await log_message(f"Starting concurrent processing of all images...")
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    all_results = {filename: result for filename, result in results}
    
    json_count = sum(1 for result in all_results.values() 
                      if result.get("status") == "success" and result.get("save_format") == "saved_as_json")
    txt_count = sum(1 for result in all_results.values() 
                     if result.get("status") == "success" and result.get("save_format") == "saved_as_txt")
    error_count = sum(1 for result in all_results.values() if result.get("status") == "error")
    skipped_count = sum(1 for result in all_results.values() if result.get("status") == "skipped")
    
    total_time = end_time - start_time
    avg_time = total_time / (len(image_files) - skipped_count) if (len(image_files) - skipped_count) > 0 else 0
    
    await log_message(f"\nAnalysis Summary:")
    await log_message(f"Total images: {len(all_results)}")
    await log_message(f"JSON format: {json_count}")
    await log_message(f"TXT format: {txt_count}")
    await log_message(f"Errors: {error_count}")
    await log_message(f"Skipped: {skipped_count}")
    await log_message(f"Total processing time: {total_time:.2f}s")
    await log_message(f"Average processing time per image: {avg_time:.2f}s")
    
    with open(os.path.join(OUTPUT_DIR, "analysis_summary.json"), "w", encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": len(all_results),
                "json_count": json_count,
                "txt_count": txt_count,
                "error_count": error_count,
                "skipped_count": skipped_count,
                "total_processing_time": total_time,
                "average_processing_time": avg_time
            },
            "details": all_results
        }, f, ensure_ascii=False, indent=2)
    
    await log_message(f"Results saved to {OUTPUT_DIR}")
    await log_message(f"Processing completed.")

if __name__ == "__main__":
    asyncio.run(main())
