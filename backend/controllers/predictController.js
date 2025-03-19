const axios = require('axios');

exports.getPrediction = async (req, res) => {
    try {
        const formData = req.body;
        console.log("Received Data:", formData);

        // Ensure all fields are present
        if (!formData.Fever || !formData.Cough || !formData.Age || !formData.Gender) {
            return res.status(400).json({ error: "Missing required fields" });
        }

        // Send data to ML Model (Python Flask Server)
        const response = await axios.post("http://127.0.0.1:5000/predict", formData);
        return res.json(response.data);
    } catch (error) {
        console.error("🔥 Error in Node.js:", error.message);
        console.error("🔥 Full Error Object:", error);

        return res.status(500).json({ error: "Server error in Node.js", details: error.message });
    }
};
