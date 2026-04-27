package com.criminal;

/**
 * =============================================================================
 * Criminal Face Detection System — Main Entry Point
 * =============================================================================
 *
 * This is the entry point for the Java frontend application.
 * It performs the following initialization steps:
 *   1. Load the OpenCV native library (required for camera access)
 *   2. Launch the Swing GUI on the Event Dispatch Thread (EDT)
 *
 * Prerequisites:
 *   - Python backend must be running at http://localhost:8000
 *   - Webcam must be connected and accessible
 *
 * Usage:
 *   mvn clean package
 *   java -jar target/criminal-face-detection-1.0.jar
 */

import com.criminal.gui.MainFrame;
import com.criminal.config.AppConfig;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;

public class Main {

    public static void main(String[] args) {
        System.out.println("==============================================");
        System.out.println("  Criminal Face Detection System — Frontend");
        System.out.println("==============================================");

        // -----------------------------------------------------------------
        // Step 1: Load OpenCV native library
        // The org.openpnp:opencv Maven package includes native libraries
        // for all major platforms. nu.pattern.OpenCV.loadLocally() extracts
        // and loads the correct native library for the current OS.
        // -----------------------------------------------------------------
        try {
            nu.pattern.OpenCV.loadLocally();
            System.out.println("[OK] OpenCV native library loaded successfully.");
        } catch (Exception e) {
            System.err.println("[FATAL] Failed to load OpenCV native library!");
            System.err.println("  Error: " + e.getMessage());
            System.err.println("  Make sure 'org.openpnp:opencv' dependency is in pom.xml.");
            System.exit(1);
        }

        // -----------------------------------------------------------------
        // Step 2: Print resolved configuration for debugging
        // -----------------------------------------------------------------
        AppConfig.printConfig();

        // -----------------------------------------------------------------
        // Step 3: Set the system look-and-feel for native appearance
        // -----------------------------------------------------------------
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            System.err.println("[WARN] Could not set system look-and-feel: " + e.getMessage());
        }

        // -----------------------------------------------------------------
        // Step 4: Launch the GUI on the Swing Event Dispatch Thread (EDT)
        // All Swing operations must happen on the EDT to avoid threading
        // issues. SwingUtilities.invokeLater ensures this.
        // -----------------------------------------------------------------
        SwingUtilities.invokeLater(() -> {
            System.out.println("[OK] Launching GUI...");
            MainFrame frame = new MainFrame();
            frame.setVisible(true);
        });
    }
}
