package com.criminal.config;

/**
 * =============================================================================
 * Application Configuration
 * =============================================================================
 *
 * Centralized configuration for the Java frontend application.
 * All configurable parameters are defined here with sensible defaults.
 *
 * To override defaults, set the corresponding system property or
 * environment variable before launching the application:
 *
 *   java -Dbackend.url=http://192.168.1.100:8000 -jar criminal-face-detection-1.0.jar
 *
 * Or set environment variables:
 *   set BACKEND_URL=http://192.168.1.100:8000
 */
public class AppConfig {

    // =========================================================================
    // Backend Connection
    // =========================================================================

    /** URL of the Python FastAPI backend server. */
    public static final String BACKEND_URL = getProperty(
            "backend.url", "BACKEND_URL", "http://localhost:8000"
    );

    // =========================================================================
    // Webcam Settings
    // =========================================================================

    /** Camera device index (0 = default camera, 1 = secondary, etc.). */
    public static final int CAMERA_INDEX = getIntProperty(
            "camera.index", "CAMERA_INDEX", 0
    );

    /** Target frames per second for webcam capture. */
    public static final int CAPTURE_FPS = getIntProperty(
            "capture.fps", "CAPTURE_FPS", 15
    );

    /** Webcam capture width in pixels. */
    public static final int CAPTURE_WIDTH = getIntProperty(
            "capture.width", "CAPTURE_WIDTH", 640
    );

    /** Webcam capture height in pixels. */
    public static final int CAPTURE_HEIGHT = getIntProperty(
            "capture.height", "CAPTURE_HEIGHT", 480
    );

    // =========================================================================
    // Recognition Settings
    // =========================================================================

    /**
     * Interval (in milliseconds) between recognition API calls.
     * Lower values = more responsive but higher CPU/network load.
     * Recommended: 300–1000ms.
     */
    public static final int RECOGNITION_INTERVAL_MS = getIntProperty(
            "recognition.interval", "RECOGNITION_INTERVAL_MS", 500
    );

    /**
     * Interval (in milliseconds) between GUI display refreshes.
     * 33ms ≈ 30 FPS display refresh rate.
     */
    public static final int DISPLAY_REFRESH_MS = getIntProperty(
            "display.refresh", "DISPLAY_REFRESH_MS", 33
    );

    // =========================================================================
    // GUI Settings
    // =========================================================================

    /** Main window width in pixels. */
    public static final int WINDOW_WIDTH = getIntProperty(
            "window.width", "WINDOW_WIDTH", 1100
    );

    /** Main window height in pixels. */
    public static final int WINDOW_HEIGHT = getIntProperty(
            "window.height", "WINDOW_HEIGHT", 700
    );

    /** Width of the alerts panel (right side) in pixels. */
    public static final int ALERTS_PANEL_WIDTH = getIntProperty(
            "alerts.panel.width", "ALERTS_PANEL_WIDTH", 320
    );

    /** Maximum number of alerts to display in the GUI. */
    public static final int MAX_GUI_ALERTS = getIntProperty(
            "alerts.max", "MAX_GUI_ALERTS", 100
    );

    // =========================================================================
    // API Timeouts
    // =========================================================================

    /** HTTP connection timeout in seconds. */
    public static final int HTTP_CONNECT_TIMEOUT = getIntProperty(
            "http.connect.timeout", "HTTP_CONNECT_TIMEOUT", 10
    );

    /** HTTP request timeout in seconds. */
    public static final int HTTP_REQUEST_TIMEOUT = getIntProperty(
            "http.request.timeout", "HTTP_REQUEST_TIMEOUT", 30
    );

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /**
     * Get a string property, checking system properties first, then
     * environment variables, then falling back to the default.
     *
     * @param sysProp   System property name (e.g., "backend.url").
     * @param envVar    Environment variable name (e.g., "BACKEND_URL").
     * @param defaultVal Default value if neither is set.
     * @return The resolved property value.
     */
    private static String getProperty(String sysProp, String envVar, String defaultVal) {
        // Check system property first (-Dkey=value)
        String value = System.getProperty(sysProp);
        if (value != null && !value.isEmpty()) {
            return value;
        }

        // Check environment variable
        value = System.getenv(envVar);
        if (value != null && !value.isEmpty()) {
            return value;
        }

        // Use default
        return defaultVal;
    }

    /**
     * Get an integer property with the same resolution order.
     */
    private static int getIntProperty(String sysProp, String envVar, int defaultVal) {
        String value = getProperty(sysProp, envVar, null);
        if (value != null) {
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException e) {
                System.err.println("[WARN] Invalid integer for " + sysProp + ": " + value);
            }
        }
        return defaultVal;
    }

    /**
     * Print all current configuration values (for debugging).
     */
    public static void printConfig() {
        System.out.println("┌──────────────────────────────────────────────────┐");
        System.out.println("│            Application Configuration             │");
        System.out.println("├──────────────────────────────────────────────────┤");
        System.out.printf("│  Backend URL:          %-25s │%n", BACKEND_URL);
        System.out.printf("│  Camera Index:         %-25d │%n", CAMERA_INDEX);
        System.out.printf("│  Capture FPS:          %-25d │%n", CAPTURE_FPS);
        System.out.printf("│  Capture Resolution:   %-25s │%n", CAPTURE_WIDTH + "x" + CAPTURE_HEIGHT);
        System.out.printf("│  Recognition Interval: %-22s ms │%n", RECOGNITION_INTERVAL_MS);
        System.out.printf("│  Display Refresh:      %-22s ms │%n", DISPLAY_REFRESH_MS);
        System.out.printf("│  Window Size:          %-25s │%n", WINDOW_WIDTH + "x" + WINDOW_HEIGHT);
        System.out.println("└──────────────────────────────────────────────────┘");
    }
}
