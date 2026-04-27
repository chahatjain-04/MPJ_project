package com.criminal.api;

/**
 * =============================================================================
 * REST API Client
 * =============================================================================
 *
 * Communicates with the Python FastAPI backend using Java's built-in
 * java.net.http.HttpClient (Java 11+). No external HTTP libraries needed.
 *
 * Endpoints used:
 *   POST /detect       → Send image, get face bounding boxes
 *   POST /recognize    → Send image, get identity + confidence
 *   POST /add-criminal → Register a new criminal with face image
 *   GET  /alerts       → Get recent detection alerts
 *
 * All image data is sent as base64-encoded JPEG strings within JSON payloads.
 * Responses are parsed using Gson.
 */

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.criminal.config.AppConfig;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

public class ApiClient {

    // Backend server URL (default: localhost:8000)
    private final String baseUrl;

    // Java HTTP client (supports HTTP/2 and connection pooling)
    private final HttpClient httpClient;

    // JSON serializer/deserializer
    private final Gson gson;

    /**
     * Create an API client pointing to the default backend URL.
     */
    public ApiClient() {
        this(AppConfig.BACKEND_URL);
    }

    /**
     * Create an API client pointing to a custom backend URL.
     *
     * @param baseUrl The backend URL (e.g., "http://192.168.1.100:8000").
     */
    public ApiClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.gson = new Gson();

        // Build HTTP client with connection timeout and HTTP/1.1
        // (HTTP/2 can cause issues with some setups)
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(AppConfig.HTTP_CONNECT_TIMEOUT))
                .version(HttpClient.Version.HTTP_1_1)
                .build();

        System.out.println("[OK] API client initialized. Backend URL: " + baseUrl);
    }

    // =========================================================================
    // POST /detect — Face detection only
    // =========================================================================

    /**
     * Send an image to the backend for face detection.
     *
     * @param base64Image Base64-encoded JPEG image.
     * @return DetectResponse with list of bounding boxes.
     * @throws Exception if the request fails.
     */
    public DetectResponse detect(String base64Image) throws Exception {
        // Build JSON payload: { "image": "<base64 data>" }
        JsonObject payload = new JsonObject();
        payload.addProperty("image", base64Image);

        // Send POST request
        String responseBody = postRequest("/detect", payload.toString());

        // Parse response
        return parseDetectResponse(responseBody);
    }

    // =========================================================================
    // POST /recognize — Face detection + identification
    // =========================================================================

    /**
     * Send an image to the backend for face recognition.
     * This is the primary method used during real-time monitoring.
     *
     * @param base64Image Base64-encoded JPEG image.
     * @return RecognizeResponse with identity, confidence, and disguise status.
     * @throws Exception if the request fails.
     */
    public RecognizeResponse recognize(String base64Image) throws Exception {
        JsonObject payload = new JsonObject();
        payload.addProperty("image", base64Image);

        String responseBody = postRequest("/recognize", payload.toString());

        return parseRecognizeResponse(responseBody);
    }

    // =========================================================================
    // POST /add-criminal — Register a new criminal
    // =========================================================================

    /**
     * Add a new criminal to the database.
     *
     * @param name        Criminal's name.
     * @param crime       Description of the crime.
     * @param base64Image Base64-encoded face image.
     * @return AddCriminalResponse with success status and message.
     * @throws Exception if the request fails.
     */
    public AddCriminalResponse addCriminal(String name, String crime, String base64Image) throws Exception {
        JsonObject payload = new JsonObject();
        payload.addProperty("name", name);
        payload.addProperty("crime", crime);
        payload.addProperty("image", base64Image);

        String responseBody = postRequest("/add-criminal", payload.toString());

        JsonObject json = JsonParser.parseString(responseBody).getAsJsonObject();
        AddCriminalResponse response = new AddCriminalResponse();
        response.success = json.get("success").getAsBoolean();
        response.message = json.get("message").getAsString();
        if (json.has("criminal_id") && !json.get("criminal_id").isJsonNull()) {
            response.criminalId = json.get("criminal_id").getAsInt();
        }
        return response;
    }

    // =========================================================================
    // GET /alerts — Recent detection alerts
    // =========================================================================

    /**
     * Retrieve recent detection alerts from the backend.
     *
     * @param limit Maximum number of alerts to retrieve.
     * @return List of AlertInfo objects.
     * @throws Exception if the request fails.
     */
    public List<AlertInfo> getAlerts(int limit) throws Exception {
        String responseBody = getRequest("/alerts?limit=" + limit);

        JsonObject json = JsonParser.parseString(responseBody).getAsJsonObject();
        JsonArray alertsArray = json.getAsJsonArray("alerts");

        List<AlertInfo> alerts = new ArrayList<>();
        for (JsonElement element : alertsArray) {
            JsonObject alertJson = element.getAsJsonObject();
            AlertInfo alert = new AlertInfo();
            alert.id = alertJson.get("id").getAsInt();
            alert.criminalName = alertJson.has("criminal_name") && !alertJson.get("criminal_name").isJsonNull()
                    ? alertJson.get("criminal_name").getAsString()
                    : "Unknown";
            alert.confidence = alertJson.get("confidence").getAsDouble();
            alert.isDisguised = alertJson.get("is_disguised").getAsBoolean();
            alert.detectedAt = alertJson.get("detected_at").getAsString();
            alerts.add(alert);
        }
        return alerts;
    }

    // =========================================================================
    // HTTP Request Helpers
    // =========================================================================

    /**
     * Send a POST request with JSON body.
     *
     * @param endpoint API endpoint path (e.g., "/detect").
     * @param jsonBody JSON string payload.
     * @return Response body as a string.
     * @throws Exception if the request fails or returns non-200.
     */
    private String postRequest(String endpoint, String jsonBody) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + endpoint))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(AppConfig.HTTP_REQUEST_TIMEOUT))
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();

        HttpResponse<String> response = httpClient.send(
                request, HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() != 200) {
            throw new RuntimeException(
                    "API error " + response.statusCode() + ": " + response.body()
            );
        }

        return response.body();
    }

    /**
     * Send a GET request.
     *
     * @param endpoint API endpoint path with query parameters.
     * @return Response body as a string.
     * @throws Exception if the request fails or returns non-200.
     */
    private String getRequest(String endpoint) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + endpoint))
                .header("Accept", "application/json")
                .timeout(Duration.ofSeconds(AppConfig.HTTP_REQUEST_TIMEOUT))
                .GET()
                .build();

        HttpResponse<String> response = httpClient.send(
                request, HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() != 200) {
            throw new RuntimeException(
                    "API error " + response.statusCode() + ": " + response.body()
            );
        }

        return response.body();
    }

    // =========================================================================
    // Response Parsing Helpers
    // =========================================================================

    private DetectResponse parseDetectResponse(String json) {
        JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
        DetectResponse response = new DetectResponse();
        response.count = obj.get("count").getAsInt();
        response.faces = new ArrayList<>();

        JsonArray facesArray = obj.getAsJsonArray("faces");
        for (JsonElement el : facesArray) {
            JsonObject faceJson = el.getAsJsonObject();
            BoundingBox box = new BoundingBox();
            box.x = faceJson.get("x").getAsInt();
            box.y = faceJson.get("y").getAsInt();
            box.width = faceJson.get("width").getAsInt();
            box.height = faceJson.get("height").getAsInt();
            box.confidence = faceJson.get("confidence").getAsDouble();
            response.faces.add(box);
        }
        return response;
    }

    private RecognizeResponse parseRecognizeResponse(String json) {
        JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
        RecognizeResponse response = new RecognizeResponse();
        response.results = new ArrayList<>();

        JsonArray resultsArray = obj.getAsJsonArray("results");
        for (JsonElement el : resultsArray) {
            JsonObject resultJson = el.getAsJsonObject();
            RecognitionResult result = new RecognitionResult();
            result.name = resultJson.get("name").getAsString();
            result.confidence = resultJson.get("confidence").getAsDouble();
            result.isDisguised = resultJson.get("is_disguised").getAsBoolean();

            // label: "criminal" → backend confirmed a DB match
            //        "unknown"  → no match found
            result.label = resultJson.has("label")
                    ? resultJson.get("label").getAsString()
                    : "unknown";

            JsonObject boxJson = resultJson.getAsJsonObject("bounding_box");
            result.boundingBox = new BoundingBox();
            result.boundingBox.x = boxJson.get("x").getAsInt();
            result.boundingBox.y = boxJson.get("y").getAsInt();
            result.boundingBox.width = boxJson.get("width").getAsInt();
            result.boundingBox.height = boxJson.get("height").getAsInt();
            result.boundingBox.confidence = boxJson.get("confidence").getAsDouble();

            // -----------------------------------------------------------------
            // Match level classification based on label + confidence.
            // Use the backend label field ("criminal"/"unknown") instead of
            // parsing the name string so disguise suffixes don't break logic.
            // -----------------------------------------------------------------
            boolean isCriminal = "criminal".equals(result.label);
            double confPercent  = result.confidence * 100.0;

            if (isCriminal && confPercent > 75.0) {
                result.matchLevel = "Strong Match";
            } else if (isCriminal) {
                result.matchLevel = "Possible Match";
            } else {
                result.matchLevel = "Unknown";
            }

            response.results.add(result);
        }
        return response;
    }

    // =========================================================================
    // Data Transfer Objects (nested classes for response data)
    // =========================================================================

    /** Bounding box for a detected face. */
    public static class BoundingBox {
        public int x, y, width, height;
        public double confidence;
    }

    /** Response from /detect endpoint. */
    public static class DetectResponse {
        public List<BoundingBox> faces;
        public int count;
    }

    /** Single recognition result for one face. */
    public static class RecognitionResult {
        public String name;
        /** "criminal" if matched in the DB, "unknown" otherwise. */
        public String label;
        public double confidence;
        public boolean isDisguised;
        public BoundingBox boundingBox;

        /**
         * Match level derived from label + confidence:
         *   - "Strong Match"   → criminal AND confidence > 75%
         *   - "Possible Match" → criminal AND confidence ≤ 75%
         *   - "Unknown"        → label is "unknown"
         */
        public String matchLevel;

        /** Convenience: true when this face matched a criminal in the DB. */
        public boolean isCriminal() {
            return "criminal".equals(label);
        }
    }

    /** Response from /recognize endpoint. */
    public static class RecognizeResponse {
        public List<RecognitionResult> results;
    }

    /** Response from /add-criminal endpoint. */
    public static class AddCriminalResponse {
        public boolean success;
        public String message;
        public int criminalId;
    }

    /** Alert information from /alerts endpoint. */
    public static class AlertInfo {
        public int id;
        public String criminalName;
        public double confidence;
        public boolean isDisguised;
        public String detectedAt;

        @Override
        public String toString() {
            return String.format("[%s] %s (%.1f%%) %s",
                    detectedAt,
                    criminalName,
                    confidence * 100,
                    isDisguised ? "[DISGUISED]" : "");
        }
    }
}
