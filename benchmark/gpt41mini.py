import os
import asyncio
import base64
import re
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI(api_key="") 

STANDARD_PROMPT = """CRITICAL INSTRUCTION: This is a geolocation prediction test where you MUST output coordinates for EVERY image.

Analyze the image and make your BEST GUESS of its geographic location based on visible clues:
- Architecture and buildings
- Vegetation and landscape
- Road features and infrastructure
- Vehicles and people
- Signs, text or other cultural indicators

After your analysis, you MUST end with coordinates in EXACTLY this format:

<answer>
lat: [latitude with 5 decimal places]
lon: [longitude with 5 decimal places]
</answer>

This is a TEST of your ability to make predictions with limited information. Refusing to provide coordinates is NOT an option. If uncertain, make your best educated guess.

WARNING: Your response MUST include coordinates in the exact format above, even if you are uncertain."""

async def process_image(image_path, image_id):

    try:

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        response = await client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": STANDARD_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"  # High detail analysis
                            }
                        }
                    ]
                }
            ],
        )
        
        # Extract response
        result_text = response.choices[0].message.content
        
        # Extract coordinates
        pattern = r"<answer>\s*lat\s*:\s*(-?\d+\.?\d*)\s*lon\s*:\s*(-?\d+\.?\d*)\s*</answer>"
        match = re.search(pattern, result_text, re.DOTALL)
        
        if match:
            result = {
                "pred_lat": float(match.group(1)),
                "pred_lon": float(match.group(2)),
                "full_response": result_text,
                "image_id": image_id
            }
            return result
        else:
            return {"image_id": image_id, "pred_lat": None, "pred_lon": None}
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return {"image_id": image_id, "pred_lat": None, "pred_lon": None}

async def main():
    # Configuration
    image_dir = "test_LLM"
    metadata_csv = "LLM_test.csv"
    output_csv = "4.1-mini.csv"
    concurrency_limit = 10 
  
    true_coords = {}
    if os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
        for _, row in df.iterrows():
            true_coords[row["image_id"]] = {
                "true_lat": row["lat"],
                "true_lon": row["lon"],
                "benchmark": row["benchmark"]
            }

    image_files = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append((os.path.join(image_dir, file), file))
    
    # Check for existing results to avoid reprocessing
    processed_images = set()
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_images = set(existing_df["image_id"].tolist())
    
    # Filter out already processed images
    image_files = [(path, img_id) for path, img_id in image_files if img_id not in processed_images]
    print(f"Processing {len(image_files)} new images")
    
    # Create semaphore to control concurrency
    sem = asyncio.Semaphore(concurrency_limit)
    
    async def process_with_semaphore(img_path, img_id):
        async with sem:
            return await process_image(img_path, img_id)

    tasks = [process_with_semaphore(path, img_id) for path, img_id in image_files]
  
    api_calls = 0
    batch_number = 0
  
    new_results = []
    
    # Execute tasks
    for task in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        result = await task
        new_results.append(result)
        
        api_calls += 1

        if api_calls % 10 == 0:
            print(f"Pausing for 60 seconds after {api_calls} API calls...")
            await asyncio.sleep(60)
        
      
        if len(new_results) % 50 == 0:
            batch_number += 1
            current_batch = new_results[-50:]
            batch_df = pd.DataFrame(current_batch)
   
            for i, row in batch_df.iterrows():
                img_id = row["image_id"]
                if img_id in true_coords:
                    batch_df.at[i, "true_lat"] = true_coords[img_id]["true_lat"]
                    batch_df.at[i, "true_lon"] = true_coords[img_id]["true_lon"]
                    batch_df.at[i, "benchmark"] = true_coords[img_id]["benchmark"]

            if os.path.exists(output_csv):
                existing_df = pd.read_csv(output_csv)
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                combined_df.to_csv(output_csv, index=False)
            else:
                batch_df.to_csv(output_csv, index=False)
                
            print(f"Saved batch #{batch_number} with {len(batch_df)} images (Total processed: {len(new_results)}, API calls: {api_calls})")
    
    remaining = len(new_results) % 50
    if remaining > 0:
        current_batch = new_results[-remaining:]
        batch_df = pd.DataFrame(current_batch)

        for i, row in batch_df.iterrows():
            img_id = row["image_id"]
            if img_id in true_coords:
                batch_df.at[i, "true_lat"] = true_coords[img_id]["true_lat"]
                batch_df.at[i, "true_lon"] = true_coords[img_id]["true_lon"]
                batch_df.at[i, "benchmark"] = true_coords[img_id]["benchmark"]
        
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
            combined_df.to_csv(output_csv, index=False)
        else:
            batch_df.to_csv(output_csv, index=False)
            
        print(f"Saved final batch with {remaining} images")
    
    print(f"Processing complete! Total images processed: {len(new_results)}, Total API calls: {api_calls}")

if __name__ == "__main__":
    asyncio.run(main())
