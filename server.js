const express = require('express');
const mongoose = require('mongoose');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

const app = express();

// Ensure uploads directory exists
const uploadDir = 'uploads';
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}
const PORT = process.env.PORT || 3000;

// MongoDB connection
mongoose.connect('mongodb://localhost:27017/endoskin', {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => console.log('Connected to MongoDB'))
.catch(err => console.error('MongoDB connection error:', err));

// Define AnalysisResult schema
const analysisResultSchema = new mongoose.Schema({
    imagePath: String,
    results: Object,
    createdAt: { type: Date, default: Date.now }
});
const AnalysisResult = mongoose.model('AnalysisResult', analysisResultSchema);

// Multer configuration for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});
const upload = multer({ storage });

// Middleware
app.use(express.json());
app.use(express.static(__dirname));

// API Endpoints
app.post('/api/analyze', upload.single('image'), async (req, res) => {
    try {
        // Load and process image with incepti.h5 model
        const modelPath = path.join(__dirname, 'incepti.h5');
        if (!fs.existsSync(modelPath)) {
            throw new Error('Model file incepti.h5 not found');
        }

        const model = await tf.loadLayersModel(`file://${modelPath}`);
        
        // Read and preprocess image
        const imageBuffer = fs.readFileSync(req.file.path);
        let imageTensor = tf.node.decodeImage(imageBuffer, 3);
        imageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);
        imageTensor = imageTensor.div(255.0).expandDims();
        
        // Make prediction
        const predictions = model.predict(imageTensor);
        const results = await predictions.array();
        
        const modelResults = {
            predictions: results[0],
            processedAt: new Date().toISOString()
        };

        // Save to MongoDB
        const result = new AnalysisResult({
            imagePath: req.file.path,
            results: modelResults
        });
        await result.save();

        res.json({
            success: true,
            results: modelResults,
            imageUrl: `/uploads/${req.file.filename}`
        });
    } catch (error) {
        console.error('Error processing image:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
