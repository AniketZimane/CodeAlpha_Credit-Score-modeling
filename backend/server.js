const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();
const port = 3001;

app.use(cors()); // Enable CORS for frontend requests
app.use(bodyParser.json()); // Parse JSON request bodies

// Import ML model route
const predictRouter = require("./routes/predict");
app.use("/predict", predictRouter);

// Global error handling
app.use((err, req, res, next) => {
  console.error("Internal Server Error:", err);
  res.status(500).json({ error: "Internal Server Error", details: err.message });
});

// Start server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
