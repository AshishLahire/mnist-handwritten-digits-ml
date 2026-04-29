from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and PCA
model = None
pca = None

def train_model():
    global model, pca
    try:
        logger.info("Training Classical ML Ensemble (KNN + SVM + DT) on FULL MNIST...")
        # Load the full dataset
        if not os.path.exists("mnist_train.csv"):
            logger.error("mnist_train.csv not found! Model will not be trained.")
            return

        train_df = pd.read_csv("mnist_train.csv")
        
        # Deployment RAM Safety: Use subset if on cloud (Render/Heroku)
        if os.environ.get('PORT') or os.environ.get('RENDER'):
            logger.info("Cloud environment detected. Using 20k samples to stay within RAM limits.")
            train_df = train_df.sample(n=min(20000, len(train_df)), random_state=42)

        X = train_df.drop("label", axis=1).values / 255.0
        y = train_df["label"].values
        
        # PCA reduction (Recommended in Task 2.3)
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X)
        
        # 1. K-Nearest Neighbors (Tuned K=3)
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=3)
        
        # 2. SVM (RBF Kernel, Tuned C and Gamma)
        from sklearn.svm import SVC
        svm = SVC(kernel="rbf", C=5, gamma="scale", probability=True)
        
        # 3. Decision Tree (Tuned Max Depth)
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_depth=15, random_state=42)
        
        # Bonus Task: Implement a Voting Ensemble
        from sklearn.ensemble import VotingClassifier
        model = VotingClassifier(
            estimators=[('knn', knn), ('svm', svm), ('dt', dt)],
            voting='soft' # Better for confidence scores
        )
        
        model.fit(X_pca, y)
        logger.info("Classical Ensemble training complete. Optimized for highest possible accuracy.")
    except Exception as e:
        logger.error(f"Error training model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 503
    
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Convert to numpy and reshape
        img_array = np.array(data).reshape(1, -1)
        
        # Use PCA as per Task 2.3
        img_pca = pca.transform(img_array)
        
        # Predict using the Classical Ensemble
        prediction = model.predict(img_pca)[0]
        probabilities = model.predict_proba(img_pca)[0]
        confidence = float(np.max(probabilities))
        
        logger.info(f"Ensemble Predicted: {prediction} with confidence: {confidence:.4f}")
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': confidence
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    train_model()
    # Use environment port for deployment (Render/Heroku)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
