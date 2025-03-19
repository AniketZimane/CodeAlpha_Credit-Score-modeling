const express = require("express");
const router = express.Router();
const axios = require("axios");

// Define the prediction route
router.post("/", async (req, res) => {
  try {
    console.log("Received request body:", req.body); // Debugging log

    const { Fever, Cough, Fatigue, Age, Gender } = req.body;

    if (!Fever || !Cough || !Fatigue || !Age || !Gender) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    // Send data to ML model (Flask)
    const mlResponse = await axios.post("http://127.0.0.1:5000/predict", req.body);

    console.log("ML Response:", mlResponse.data); // Debugging log
    res.json(mlResponse.data);
  } catch (error) {
    console.error("Error in prediction route:", error.message);
    res.status(500).json({ error: "Server error in Node.js", details: error.message });
  }
});

module.exports = router;
