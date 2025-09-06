
import os
import json
import glob
import shutil
from tqdm import tqdm
import argparse

def setup_argparse():
    parser = argparse.ArgumentParser(description='Convert HFRL dataset to LLaMA Factory format')
    parser.add_argument('--input_dir', type=str, default="/home/ubuntu/HFRL",
                        help='Input directory containing original JSON and images')
    parser.add_argument('--output_dir', type=str, default="./data",
                        help='Output directory for processed data')
    parser.add_argument('--image_token', type=str, default="<image>",
                        choices=["<image>", "<image_soft_token>"],
                        help='Image token to use')
    parser.add_argument('--dataset_name', type=str, default="hfrl_data",
                        help='Dataset name')
    return parser.parse_args()

def create_directories(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

def generate_instruction_template(image_token):
    return f"""{image_token}

## Overview
I will provide you with an image of a location. Your task is to identify the corresponding longitude and latitude, which requires extremely detailed geographic analysis, identifying microscopic features that help determine the precise location. This requires a meticulous examination of all visual elements from both a macro and micro perspective.
## Required JSON Output Format
Your analysis must be provided in a structured JSON format as follows:
```json
{{
  "image_id": "filename.jpg",
  "thinking": {{
    "25km_analysis": {{
      "terrain_characteristics": "Detailed description of regional terrain",
      "climate_features": "Climate zone, seasonal indicators, weather patterns",
      "regional_development": "Historical context, development patterns, land use"
    }},
    "1km_analysis": {{
      "road_engineering": {{
        "surface_materials": "Precise description of road material composition, texture, color, wear patterns, age indicators",
        "line_markings": "Width, color, pattern, material, conformity to regional standards",
        "guardrails": "Design type, material, height, installation pattern, regional specificity",
        "lighting": "Pole design, height, spacing, lamp type, regional standards",
        "drainage": "Design, materials, integration with landscape, climatic adaptation"
      }},
      "vegetation": {{
        "tree_species": "Scientific names, distribution patterns, growth characteristics, management style",
        "herbaceous_plants": "Undergrowth composition, seasonal state, native vs. introduced species",
        "vegetation_management": "Pruning techniques, planting patterns, maintenance regimes"
      }}
    }},
    "buildings_infrastructure": {{
      "architectural_styles": "Building design elements, materials, roof styles, regional influences",
      "municipal_facilities": "Public infrastructure, signage systems, urban furniture design",
      "transportation_facilities": "Transit design, stops/stations, pedestrian accommodations"
    }},
    "vehicles_traffic": {{
      "vehicle_types": "Common models, regional preferences, vehicle adaptations",
      "traffic_patterns": "Flow characteristics, driver behaviors, local conventions"
    }},
    "terrain_soil": {{
      "terrain_details": "Micro-topography, slope characteristics, erosion patterns",
      "soil_characteristics": "Color, texture, composition, regional distinctiveness"
    }},
    "elimination_similar_regions": [
      {{
        "eliminated_region": "Region name",
        "reasons": [
          "Specific distinguishing feature 1 that excludes this region",
          "Specific distinguishing feature 2 that excludes this region",
          "Specific distinguishing feature 3 that excludes this region"
        ]
      }}
    ]
  }},
  "answer": {{
    "coordinates": {{
      "latitude": 00.000000,
      "longitude": 00.000000,
      "region": "Region name"
    }},
    "final_location_confirmation": {{
      "confirmed_location": "Precise description of location",
      "decisive_features": [
        "Highly distinctive feature 1 unique to this location",
        "Highly distinctive feature 2 unique to this location",
        "Highly distinctive feature 3 unique to this location"
      ],
      "accuracy_assessment": "Confidence level with justification"
    }}
  }}
}}
```

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

Finally, when submitting your assignment, be sure to verify your work to make sure you have not missed any details that could be described, or if any details do not match your description.

Your final output must be presented as properly formatted JSON according to the structure provided above. Ensure each section contains extremely detailed, specific information derived from meticulous examination of the image.

Remember to be extremely specific and detailed. Use precise measurements, exact species names, specific architectural terms, and reference regional building codes or standards where possible.
"""

def create_dataset_info(output_dir, dataset_name, image_token):
    dataset_info = {
        "alpaca_en": {
            "file_name": "alpaca_en.json"
        },
        "alpaca_en_demo": {
            "file_name": "alpaca_en_demo.json"
        },
        dataset_name: {
            "file_name": f"{dataset_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    }

    with open(os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"Created dataset_info.json file: {os.path.join(output_dir, 'dataset_info.json')}")

def convert_to_llama_factory_format(input_dir, output_dir, dataset_name, image_token):
    output_file = os.path.join(output_dir, f"{dataset_name}.json")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory {input_dir} does not exist or is not a directory. Please check the path.")
        return False

    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    if not json_files:
        print(f"Warning: No .json files found in {input_dir}.")
        return False

    dataset = []
    valid_pairs = 0
    skipped_files = 0

    instruction_template = generate_instruction_template(image_token)

    for json_file in tqdm(json_files, desc="Processing files"):
        filename = os.path.basename(json_file)
        image_id = filename.split('.')[0]
        image_filename = f"{image_id}.jpg"
        relative_image_path = image_filename

        full_image_path_check = os.path.join(input_dir, image_filename)
        if not os.path.exists(full_image_path_check):
            print(f"\nWarning: Image file does not exist {full_image_path_check}, skipping {filename}")
            skipped_files += 1
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)

            if not isinstance(original_data, dict) or "thinking" not in original_data or "answer" not in original_data:
                print(f"\nWarning: File {filename} has incorrect JSON structure, skipping.")
                skipped_files += 1
                continue

            formatted_instruction = instruction_template.replace('"image_id": "filename.jpg"', f'"image_id": "{image_filename}"')

            data_entry = {
                "images": [relative_image_path],
                "conversations": [
                    {
                        "role": "user",
                        "content": formatted_instruction
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(original_data, ensure_ascii=False, indent=2)
                    }
                ]
            }

            dataset.append(data_entry)
            valid_pairs += 1

        except json.JSONDecodeError as e:
            print(f"\nJSON parsing error while processing {filename}: {str(e)}")
            skipped_files += 1
        except KeyError as e:
            print(f"\nMissing key error while processing {filename}: {str(e)}")
            skipped_files += 1
        except Exception as e:
            print(f"\nUnknown error while processing {filename}: {str(e)}")
            skipped_files += 1

    print(f"\nSuccessfully processed {valid_pairs} valid image-JSON pairs.")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files.")

    if not dataset:
        print("No valid data entries generated, not writing output file.")
        return False

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"All processing completed! Converted {len(dataset)} data entries to {output_file}")
        return True
    except Exception as e:
        print(f"\nError writing output file {output_file}: {e}")
        return False

def generate_env_script(output_dir):
    env_script = """#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "Environment variables set for memory optimization"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
"""

    script_path = os.path.join(output_dir, "set_env.sh")
    with open(script_path, 'w') as f:
        f.write(env_script)

    os.chmod(script_path, 0o755)
    print(f"Generated memory optimization environment script: {script_path}")

def generate_training_script(output_dir, dataset_name):
    train_script = f"""#!/bin/bash
source ./data/set_env.sh

echo "Starting LLaMA Factory training..."
echo "Model: /home/ubuntu/gemma-3-12b-it"
echo "Dataset: {dataset_name}"
echo "Dataset directory: ./data"
echo "Media directory: /home/ubuntu/HFRL"
echo "Output directory: /home/ubuntu/gemma-3-12b/lora/sft"
echo "Training mode: LoRA (BF16)"
echo "---"

llamafactory-cli train \\
    --stage sft \\
    --do_train \\
    --model_name_or_path /home/ubuntu/gemma-3-27b-it \\
    --dataset {dataset_name} \\
    --dataset_dir ./data \\
    --media_dir /home/ubuntu/HFRL \\
    --template gemma3 \\
    --finetuning_type lora \\
    --output_dir ./saves/gemma-3-27b/lora/sft \\
    --overwrite_cache \\
    --overwrite_output_dir \\
    --cutoff_len 4096 \\
    --preprocessing_num_workers 16 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 8 \\
    --lr_scheduler_type cosine \\
    --logging_steps 10 \\
    --warmup_steps 20 \\
    --save_steps 100 \\
    --eval_steps 100 \\
    --evaluation_strategy steps \\
    --load_best_model_at_end \\
    --learning_rate 2e-5 \\
    --num_train_epochs 3.0 \\
    --val_size 0.05 \\
    --lora_rank 16 \\
    --lora_alpha 32 \\
    --lora_dropout 0.05 \\
    --report_to none \\
    --flash_attn auto \\
    --bf16 true

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "---"
  echo "Training script completed successfully."
else
  echo "---"
  echo "Training script failed with exit code: $EXIT_CODE" >&2
  exit $EXIT_CODE
fi

exit 0
"""

    script_path = os.path.join(output_dir, "train.sh")
    with open(script_path, 'w') as f:
        f.write(train_script)

    os.chmod(script_path, 0o755)
    print(f"Generated training script: {script_path}")

def main():
    args = setup_argparse()

    print(f"===== Geographic Location Dataset Conversion Tool =====")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Image token: {args.image_token}")
    print(f"===============================")

    create_directories(args.output_dir)

    create_dataset_info(args.output_dir, args.dataset_name, args.image_token)

    success = convert_to_llama_factory_format(
        args.input_dir,
        args.output_dir,
        args.dataset_name,
        args.image_token
    )

    if success:
        generate_env_script(args.output_dir)

        generate_training_script(args.output_dir, args.dataset_name)

        print("\n=== Complete! ===")
        print("All files have been converted and are ready.")
        print(f"1. Run 'source {args.output_dir}/set_env.sh' to set environment variables")
        print(f"2. Run 'bash {args.output_dir}/train.sh' to start training")
        print("Note: Training script is optimized to reduce OOM risk with reduced LoRA parameters and optimized memory usage")
    else:
        print("\nConversion failed, please check errors and retry.")

if __name__ == "__main__":
    main()
