import pandas as pd
import joblib
import os

def main():
    # Path configuration
    base_dir = '/Users/nipunagarwal/Desktop/Worley data/L5 impute/Model 3'
    model_path = os.path.join(base_dir, 'unspsc_model.joblib')
    test_path = os.path.join(base_dir, 'L5 test set.xlsx')
    output_path = os.path.join(base_dir, 'predictions.xlsx')

    # Load model and data
    classifier = joblib.load(model_path)
    test_df = pd.read_excel(test_path)
    
    # Predict
    test_df['processed_text'] = test_df.apply(
        lambda row: classifier.preprocess_text(f"{row['supplier']} {row['item_description']}"), 
        axis=1
    )
    test_df['Predicted_L5'] = classifier.predict(test_df['processed_text'])
    
    # Save results
    test_df[['item_description', 'Predicted_L5']].to_excel(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
