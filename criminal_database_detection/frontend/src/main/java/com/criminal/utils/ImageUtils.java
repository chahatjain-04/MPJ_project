package com.criminal.utils;

/**
 * =============================================================================
 * Image Utility Methods
 * =============================================================================
 *
 * Helper methods for image conversion, encoding, and manipulation
 * used across the Java frontend application.
 *
 * Conversions supported:
 *   - OpenCV Mat → BufferedImage
 *   - BufferedImage → OpenCV Mat
 *   - OpenCV Mat → Base64 JPEG string
 *   - File → Base64 string
 *   - BufferedImage resize
 */

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Base64;
import javax.imageio.ImageIO;

public class ImageUtils {

    // Private constructor — utility class, no instantiation
    private ImageUtils() {}

    // =========================================================================
    // OpenCV Mat ↔ BufferedImage Conversions
    // =========================================================================

    /**
     * Convert an OpenCV Mat to a Java BufferedImage.
     *
     * Handles both grayscale (1-channel) and color (3-channel BGR) images.
     * OpenCV uses BGR channel order; BufferedImage TYPE_3BYTE_BGR also uses BGR,
     * so no color conversion is needed.
     *
     * @param mat OpenCV Mat image (1 or 3 channels).
     * @return BufferedImage equivalent, or null if the Mat is empty.
     */
    public static BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }

        int width = mat.cols();
        int height = mat.rows();
        int channels = mat.channels();

        int imageType = (channels == 1)
                ? BufferedImage.TYPE_BYTE_GRAY
                : BufferedImage.TYPE_3BYTE_BGR;

        BufferedImage image = new BufferedImage(width, height, imageType);
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        mat.get(0, 0, pixels);

        return image;
    }

    /**
     * Convert a Java BufferedImage to an OpenCV Mat.
     *
     * @param image BufferedImage to convert.
     * @return OpenCV Mat in BGR format, or an empty Mat if the image is null.
     */
    public static Mat bufferedImageToMat(BufferedImage image) {
        if (image == null) {
            return new Mat();
        }

        // Ensure the image is in TYPE_3BYTE_BGR format
        BufferedImage converted = image;
        if (image.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            converted = new BufferedImage(
                    image.getWidth(), image.getHeight(), BufferedImage.TYPE_3BYTE_BGR
            );
            Graphics2D g2d = converted.createGraphics();
            g2d.drawImage(image, 0, 0, null);
            g2d.dispose();
        }

        byte[] pixels = ((DataBufferByte) converted.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(converted.getHeight(), converted.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, pixels);

        return mat;
    }

    // =========================================================================
    // Base64 Encoding
    // =========================================================================

    /**
     * Encode an OpenCV Mat as a base64 JPEG string.
     *
     * Used to send webcam frames to the Python backend via REST API.
     * JPEG compression provides ~10:1 compression ratio, reducing
     * network bandwidth significantly.
     *
     * @param mat OpenCV Mat image.
     * @return Base64-encoded JPEG string, or null if encoding fails.
     */
    public static String matToBase64Jpeg(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }

        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, buffer);
        byte[] jpegBytes = buffer.toArray();
        return Base64.getEncoder().encodeToString(jpegBytes);
    }

    /**
     * Encode a BufferedImage as a base64 JPEG string.
     *
     * @param image BufferedImage to encode.
     * @return Base64-encoded JPEG string, or null if encoding fails.
     */
    public static String bufferedImageToBase64Jpeg(BufferedImage image) {
        if (image == null) {
            return null;
        }

        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(image, "jpg", baos);
            byte[] jpegBytes = baos.toByteArray();
            return Base64.getEncoder().encodeToString(jpegBytes);
        } catch (IOException e) {
            System.err.println("[ERROR] Failed to encode image to base64: " + e.getMessage());
            return null;
        }
    }

    /**
     * Read a file and encode its contents as a base64 string.
     *
     * Useful for loading saved face images to send to the API.
     *
     * @param filePath Path to the image file.
     * @return Base64-encoded file contents, or null if reading fails.
     */
    public static String fileToBase64(String filePath) {
        try {
            byte[] fileBytes = Files.readAllBytes(new File(filePath).toPath());
            return Base64.getEncoder().encodeToString(fileBytes);
        } catch (IOException e) {
            System.err.println("[ERROR] Failed to read file: " + e.getMessage());
            return null;
        }
    }

    // =========================================================================
    // Image Manipulation
    // =========================================================================

    /**
     * Resize a BufferedImage to the specified dimensions.
     *
     * Uses BILINEAR interpolation for good quality at reasonable speed.
     *
     * @param original  The original image.
     * @param newWidth  Target width in pixels.
     * @param newHeight Target height in pixels.
     * @return Resized BufferedImage.
     */
    public static BufferedImage resize(BufferedImage original, int newWidth, int newHeight) {
        BufferedImage resized = new BufferedImage(newWidth, newHeight, original.getType());
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(
                java.awt.RenderingHints.KEY_INTERPOLATION,
                java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR
        );
        g2d.drawImage(original, 0, 0, newWidth, newHeight, null);
        g2d.dispose();
        return resized;
    }

    /**
     * Resize an OpenCV Mat to the specified dimensions.
     *
     * @param mat       OpenCV Mat image.
     * @param newWidth  Target width.
     * @param newHeight Target height.
     * @return Resized Mat.
     */
    public static Mat resizeMat(Mat mat, int newWidth, int newHeight) {
        Mat resized = new Mat();
        Imgproc.resize(mat, resized, new Size(newWidth, newHeight));
        return resized;
    }
}
