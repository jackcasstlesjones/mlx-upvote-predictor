"""
Main entry point for the upvote prediction application
"""
from upvote_regressor import demo, TitleVectorizer, UpvoteRegressor, train_and_save_model
import sys
import torch
import numpy as np
import os
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "hn_upvote_regressor.pth"


def predict_title_score(title, model=None, vectorizer=None):
    """
    Predict the upvote score for a given title
    
    Args:
        title: The HN post title
        model: Optional pre-trained model
        vectorizer: Optional vectorizer
        
    Returns:
        Predicted upvote score
    """
    # Initialize vectorizer if not provided
    if vectorizer is None:
        vectorizer = TitleVectorizer()
    
    # Initialize model if not provided
    if model is None:
        if os.path.exists(MODEL_PATH):
            # Load from saved model if available
            embed_dim = vectorizer.word_vectors.shape[1]
            model = UpvoteRegressor(embed_dim=embed_dim)
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
        else:
            # Train model if no saved model exists
            model = train_and_save_model(MODEL_PATH)
    
    # Get title embedding
    title_vector = vectorizer.get_title_vector(title)
    
    # Convert to tensor
    tensor = torch.tensor(title_vector, dtype=torch.float32).unsqueeze(0)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(tensor)
        # Ensure non-negative score
        score = max(0, prediction.item())
    
    return score


def main():
    """Main function to run the upvote predictor demo"""
    print("Running Hacker News Upvote Predictor Demo")
    print("=========================================")
    
    # Create model directory if it doesn't exist
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Handle training command
        if sys.argv[1] == "--train":
            print("Training model on full dataset...")
            # Always use single-stage training to train on the full dataset directly
            train_and_save_model(MODEL_PATH, force_retrain=True, two_stage=False)
            print(f"Model saved to: {MODEL_PATH}")
            return 0
            
        # Handle prediction command
        elif sys.argv[1] == "--predict":
            if len(sys.argv) > 2:
                # Get title from command line
                title = " ".join(sys.argv[2:])
                
                # Use trained model if available
                vectorizer = TitleVectorizer()
                
                # Load model (will train if necessary)
                if os.path.exists(MODEL_PATH):
                    print(f"Loading trained model from {MODEL_PATH}")
                    embed_dim = vectorizer.word_vectors.shape[1]
                    model = UpvoteRegressor(embed_dim=embed_dim)
                    model.load_state_dict(torch.load(MODEL_PATH))
                    model.eval()
                else:
                    print("No trained model found. Training new model...")
                    model = train_and_save_model(MODEL_PATH)
                
                # Predict score
                score = predict_title_score(title, model, vectorizer)
                print(f"\nTitle: {title}")
                print(f"Predicted upvotes: {score:.1f}")
                return 0
            else:
                print("Error: No title provided. Usage: python main.py --predict \"Your HN title here\"")
                return 1
    
    # Run standard demo if no special arguments
    try:
        demo(model_save_path=MODEL_PATH)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())