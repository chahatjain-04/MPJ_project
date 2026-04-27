package com.criminal.camera;

/**
 * =============================================================================
 * Webcam Capture Service
 * =============================================================================
 *
 * Captures frames from the default webcam using OpenCV's VideoCapture class.
 * Runs frame capture in a dedicated background thread to avoid blocking the
 * GUI thread.
 *
 * Features:
 *   - Captures frames at a configurable FPS (default: 15)
 *   - Converts OpenCV Mat to Java BufferedImage for GUI display
 *   - Encodes frames as base64 JPEG strings for API communication
 *   - Thread-safe frame access via synchronized methods
 *
 * Usage:
 *   WebcamCapture cam = new WebcamCapture();
 *   cam.start();
 *   BufferedImage img = cam.getBufferedImage();
 *   String base64 = cam.getBase64Frame();
 *   cam.stop();
 */

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.Base64;

public class WebcamCapture {

    // OpenCV video capture instance
    private VideoCapture capture;

    // Latest captured frame (protected by synchronized access)
    private Mat currentFrame;

    // Background thread for continuous frame capture
    private Thread captureThread;

    // Flag to control the capture loop
    private volatile boolean running = false;

    // Target capture rate
    private final int targetFPS;

    // Camera device index (0 = default camera)
    private final int cameraIndex;

    /**
     * Create a webcam capture instance with default settings.
     * Default camera (index 0), 15 FPS target rate.
     */
    public WebcamCapture() {
        this(0, 15);
    }

    /**
     * Create a webcam capture instance with custom settings.
     *
     * @param cameraIndex Index of the camera device (0 = default).
     * @param targetFPS   Target frames per second to capture.
     */
    public WebcamCapture(int cameraIndex, int targetFPS) {
        this.cameraIndex = cameraIndex;
        this.targetFPS = targetFPS;
        this.currentFrame = new Mat();
    }

    /**
     * Open the webcam and start capturing frames in a background thread.
     *
     * @return true if the camera was opened successfully, false otherwise.
     */
    public boolean start() {
        // Open the webcam
        capture = new VideoCapture(cameraIndex);

        if (!capture.isOpened()) {
            System.err.println("[ERROR] Failed to open camera at index " + cameraIndex);
            return false;
        }

        // Set capture resolution (640x480 for a good balance of quality and speed)
        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);

        System.out.println("[OK] Camera opened: " +
                (int) capture.get(Videoio.CAP_PROP_FRAME_WIDTH) + "x" +
                (int) capture.get(Videoio.CAP_PROP_FRAME_HEIGHT));

        // Start the background capture thread
        running = true;
        captureThread = new Thread(this::captureLoop, "WebcamCapture-Thread");
        captureThread.setDaemon(true);  // Thread dies when the app exits
        captureThread.start();

        return true;
    }

    /**
     * Stop capturing and release the camera.
     */
    public void stop() {
        running = false;

        // Wait for the capture thread to finish
        if (captureThread != null) {
            try {
                captureThread.join(2000);  // Wait up to 2 seconds
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        // Release the camera
        if (capture != null && capture.isOpened()) {
            capture.release();
            System.out.println("[OK] Camera released.");
        }
    }

    /**
     * Main capture loop running in the background thread.
     * Continuously reads frames from the camera at the target FPS.
     */
    private void captureLoop() {
        long frameInterval = 1000 / targetFPS;  // Milliseconds between frames
        Mat tempFrame = new Mat();

        while (running) {
            long startTime = System.currentTimeMillis();

            // Read a frame from the camera
            if (capture.read(tempFrame) && !tempFrame.empty()) {
                // Update the current frame (synchronized for thread safety)
                synchronized (this) {
                    tempFrame.copyTo(currentFrame);
                }
            } else {
                System.err.println("[WARN] Failed to read frame from camera.");
            }

            // Sleep to maintain target FPS
            long elapsed = System.currentTimeMillis() - startTime;
            long sleepTime = frameInterval - elapsed;
            if (sleepTime > 0) {
                try {
                    Thread.sleep(sleepTime);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
    }

    /**
     * Get the latest captured frame as an OpenCV Mat.
     *
     * @return A copy of the current frame, or an empty Mat if no frame yet.
     */
    public synchronized Mat getFrame() {
        Mat copy = new Mat();
        if (!currentFrame.empty()) {
            currentFrame.copyTo(copy);
        }
        return copy;
    }

    /**
     * Get the latest frame as a base64-encoded JPEG string.
     *
     * This format is used to send frames to the Python backend via REST API.
     * JPEG encoding provides good compression while maintaining quality.
     *
     * @return Base64-encoded JPEG string, or null if no frame available.
     */
    public synchronized String getBase64Frame() {
        if (currentFrame.empty()) {
            return null;
        }

        // Encode the frame as JPEG
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".jpg", currentFrame, buffer);

        // Convert to base64 string
        byte[] jpegBytes = buffer.toArray();
        return Base64.getEncoder().encodeToString(jpegBytes);
    }

    /**
     * Convert the latest OpenCV Mat frame to a Java BufferedImage.
     *
     * This conversion is necessary because Swing's JPanel uses
     * BufferedImage for rendering, not OpenCV's Mat format.
     *
     * Conversion steps:
     *   1. Get frame dimensions and channel count
     *   2. Create a BufferedImage with matching dimensions
     *   3. Copy pixel data from Mat to BufferedImage's backing array
     *
     * @return BufferedImage of the current frame, or null if no frame.
     */
    public synchronized BufferedImage getBufferedImage() {
        if (currentFrame.empty()) {
            return null;
        }

        int width = currentFrame.cols();
        int height = currentFrame.rows();
        int channels = currentFrame.channels();

        // Determine the BufferedImage type based on channel count
        // OpenCV uses BGR format; BufferedImage uses BGR for TYPE_3BYTE_BGR
        int imageType;
        if (channels == 1) {
            imageType = BufferedImage.TYPE_BYTE_GRAY;
        } else {
            imageType = BufferedImage.TYPE_3BYTE_BGR;
        }

        // Create the BufferedImage
        BufferedImage image = new BufferedImage(width, height, imageType);

        // Get the backing byte array of the BufferedImage
        byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

        // Copy pixel data from Mat to BufferedImage
        currentFrame.get(0, 0, targetPixels);

        return image;
    }

    /**
     * Check if the webcam is currently running.
     *
     * @return true if capturing frames, false otherwise.
     */
    public boolean isRunning() {
        return running;
    }
}
