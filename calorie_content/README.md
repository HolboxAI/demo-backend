# Calorie Content Analysis

This Python script performs a comprehensive analysis of food items from an image, estimates portion sizes, and calculates their calorie content based on the USDA Food Database. The process uses OpenAI's vision model to identify foods, followed by a search in the USDA FoodData Central API to retrieve nutritional data.

## Features

- **Food Identification**: Identifies distinct edible items in the uploaded image using OpenAI's vision model (gpt-4o-mini).
- **Portion Estimation**: Estimates the portion size for each food item detected in the image.
- **Calorie Estimation**: Retrieves calorie data per 100g from the USDA Food Database and calculates the estimated calories for each detected food item.
- **Comprehensive Output**: Returns a list of identified foods with their calorie information and total estimated calories.

## Requirements

- Python 3.8 or higher
- OpenAI API Key for vision model access
- USDA API Key for retrieving nutritional information

## Usage

### Functions

#### `identify_foods_with_llm(image_bytes, user_portion_hint=None, mime="image/jpeg")`
- Detects foods in the provided image and estimates their portion sizes.
- Returns a list of dictionaries containing:
  - `name`: Common name of the food item
  - `confidence`: Confidence score (0-1)
  - `portion_text`: Portion description (e.g., "1 slice", "1 bowl")
  - `estimated_grams`: Estimated weight of the portion in grams

#### `usda_search_top_hit(query)`
- Searches for a food item in the USDA Food Database and returns the most relevant result.
- Returns a dictionary containing the most relevant food item or `None` if no result is found.

#### `estimate_item_calories(item, usda_food)`
- Estimates the calories for a detected food item based on the USDA data.
- Returns a dictionary with:
  - `name`: Food item name
  - `confidence`: Confidence score
  - `portion_text`: Portion description
  - `estimated_grams`: Portion weight in grams
  - `kcal_per_100g`: Caloric content per 100g of the item
  - `estimated_calories`: Estimated calories for the portion
  - `usda_fdc_id`: USDA FoodData Central ID
  - `usda_brand`: Brand name (if available)
  - `description`: Description of the food item (if available)

#### `analyze_calories_flow(image_bytes, portion_hint=None, mime="image/jpeg")`
- Full flow: Detects foods, retrieves USDA data, and estimates calories.
- Returns a dictionary containing:
  - `items`: List of detailed food items with calorie information
  - `total_calories_estimate`: Total estimated calories for the image
