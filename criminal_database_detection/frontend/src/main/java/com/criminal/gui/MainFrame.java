package com.criminal.gui;

/**
 * =============================================================================
 * Main GUI Frame
 * =============================================================================
 *
 * The primary Swing window for the Criminal Face Detection System.
 *
 * Layout:
 *   ┌──────────────────────────────────┬─────────────────────┐
 *   │                                  │   ALERTS PANEL      │
 *   │        LIVE VIDEO FEED           │   - Recent alerts   │
 *   │   (webcam with bounding boxes,   │   - Name, time,     │
 *   │    names, and confidence scores) │     confidence      │
 *   │                                  │                     │
 *   ├──────────────────────────────────┴─────────────────────┤
 *   │  CONTROLS: [Start/Stop] [Add Criminal] [Refresh Alerts]│
 *   │  STATUS: Connected | FPS: 15 | Faces: 2               │
 *   └───────────────────────────────────────────────────────-─┘
 *
 * Features:
 *   - Real-time webcam display with overlaid bounding boxes
 *   - Color-coded boxes: RED = matched criminal, GREEN = unknown
 *   - ORANGE dashed border for disguised faces
 *   - Alerts panel showing recent detections
 *   - Control buttons for capture and criminal registration
 *   - Recognition runs every ~500ms in a background thread
 */

import com.criminal.camera.WebcamCapture;
import com.criminal.api.ApiClient;
import com.criminal.api.ApiClient.*;
import com.criminal.config.AppConfig;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class MainFrame extends JFrame {

    // --- Services ---
    private final WebcamCapture webcam;
    private final ApiClient apiClient;

    // --- GUI Components ---
    private VideoPanel videoPanel; // Custom panel for video rendering
    private JList<String> alertsList; // Scrollable alerts list
    private DefaultListModel<String> alertsModel;
    private JLabel statusLabel; // Status bar at the bottom
    private JButton startStopButton; // Toggle capture button
    private JButton addCriminalButton; // Open add-criminal dialog
    private JButton refreshAlertsButton; // Refresh the alerts list

    // --- State ---
    private boolean isCapturing = false;
    private Timer displayTimer; // GUI refresh timer (~30 FPS)
    private Timer recognitionTimer; // Recognition API call timer (~500ms)

    // Current recognition results (thread-safe list)
    private final CopyOnWriteArrayList<RecognitionResult> currentResults = new CopyOnWriteArrayList<>();

    /**
     * Construct the main application frame.
     */
    public MainFrame() {
        // Initialize services — use AppConfig for camera and backend settings
        webcam = new WebcamCapture(AppConfig.CAMERA_INDEX, AppConfig.CAPTURE_FPS);
        apiClient = new ApiClient(AppConfig.BACKEND_URL);

        // Configure the JFrame
        setTitle("Criminal Face Detection & Identification System");
        setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        setSize(AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT);
        setMinimumSize(new Dimension(900, 600));
        setLocationRelativeTo(null); // Center on screen

        // Handle window close: stop capture and release resources
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                shutdown();
                dispose();
                System.exit(0);
            }
        });

        // Build the UI
        initComponents();
        layoutComponents();

        System.out.println("[OK] GUI initialized.");
    }

    /**
     * Initialize all GUI components.
     */
    private void initComponents() {
        // --- Video Panel ---
        videoPanel = new VideoPanel();
        videoPanel.setPreferredSize(new Dimension(640, 480));

        // --- Alerts List ---
        alertsModel = new DefaultListModel<>();
        alertsModel.addElement("No alerts yet. Start capture to begin monitoring.");
        alertsList = new JList<>(alertsModel);
        alertsList.setFont(new Font("Monospaced", Font.PLAIN, 12));
        alertsList.setBackground(new Color(30, 30, 30));
        alertsList.setForeground(new Color(0, 255, 100));
        alertsList.setSelectionBackground(new Color(60, 60, 60));

        // --- Buttons ---
        startStopButton = createStyledButton("▶ Start Capture", new Color(46, 125, 50));
        startStopButton.addActionListener(e -> toggleCapture());

        addCriminalButton = createStyledButton("+ Add Criminal", new Color(21, 101, 192));
        addCriminalButton.addActionListener(e -> showAddCriminalDialog());

        refreshAlertsButton = createStyledButton("↻ Refresh Alerts", new Color(100, 100, 100));
        refreshAlertsButton.addActionListener(e -> refreshAlerts());

        // --- Status Label ---
        statusLabel = new JLabel("  Status: Ready | Backend: " + AppConfig.BACKEND_URL);
        statusLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));
        statusLabel.setForeground(Color.LIGHT_GRAY);
        statusLabel.setOpaque(true);
        statusLabel.setBackground(new Color(30, 30, 30));
        statusLabel.setBorder(new EmptyBorder(4, 8, 4, 8));
    }

    /**
     * Arrange all components in the frame using BorderLayout.
     */
    private void layoutComponents() {
        // Use a dark background for the entire frame
        getContentPane().setBackground(new Color(40, 40, 40));
        setLayout(new BorderLayout(6, 6));

        // --- Center: Video Panel ---
        JPanel videoPanelWrapper = new JPanel(new BorderLayout());
        videoPanelWrapper.setBackground(new Color(40, 40, 40));
        TitledBorder videoBorder = BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(new Color(80, 80, 80)),
                " Live Feed ",
                TitledBorder.LEFT,
                TitledBorder.TOP,
                new Font("SansSerif", Font.BOLD, 13),
                new Color(200, 200, 200));
        videoPanelWrapper.setBorder(videoBorder);
        videoPanelWrapper.add(videoPanel, BorderLayout.CENTER);

        // --- Right: Alerts Panel ---
        JPanel alertsPanel = new JPanel(new BorderLayout());
        alertsPanel.setPreferredSize(new Dimension(AppConfig.ALERTS_PANEL_WIDTH, 0));
        alertsPanel.setBackground(new Color(40, 40, 40));
        TitledBorder alertsBorder = BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(new Color(80, 80, 80)),
                " Detection Alerts ",
                TitledBorder.LEFT,
                TitledBorder.TOP,
                new Font("SansSerif", Font.BOLD, 13),
                new Color(255, 100, 100));
        alertsPanel.setBorder(alertsBorder);
        JScrollPane alertsScroll = new JScrollPane(alertsList);
        alertsScroll.setBorder(BorderFactory.createEmptyBorder());
        alertsPanel.add(alertsScroll, BorderLayout.CENTER);

        // --- Bottom: Controls + Status ---
        JPanel bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.setBackground(new Color(40, 40, 40));

        // Controls bar with buttons
        JPanel controlsPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 15, 8));
        controlsPanel.setBackground(new Color(50, 50, 50));
        controlsPanel.setBorder(BorderFactory.createMatteBorder(1, 0, 0, 0, new Color(80, 80, 80)));
        controlsPanel.add(startStopButton);
        controlsPanel.add(addCriminalButton);
        controlsPanel.add(refreshAlertsButton);

        bottomPanel.add(controlsPanel, BorderLayout.CENTER);
        bottomPanel.add(statusLabel, BorderLayout.SOUTH);

        // --- Assemble ---
        add(videoPanelWrapper, BorderLayout.CENTER);
        add(alertsPanel, BorderLayout.EAST);
        add(bottomPanel, BorderLayout.SOUTH);
    }

    /**
     * Create a styled, dark-themed button.
     */
    private JButton createStyledButton(String text, Color bgColor) {
        JButton button = new JButton(text);
        button.setFont(new Font("SansSerif", Font.BOLD, 13));
        button.setForeground(Color.WHITE);
        button.setBackground(bgColor);
        button.setFocusPainted(false);
        button.setBorderPainted(false);
        button.setCursor(new Cursor(Cursor.HAND_CURSOR));
        button.setPreferredSize(new Dimension(180, 36));
        return button;
    }

    // =========================================================================
    // Capture Control
    // =========================================================================

    /**
     * Toggle webcam capture on/off.
     */
    private void toggleCapture() {
        if (!isCapturing) {
            startCapture();
        } else {
            stopCapture();
        }
    }

    /**
     * Start the webcam capture and begin real-time recognition.
     */
    private void startCapture() {
        // Open webcam
        boolean opened = webcam.start();
        if (!opened) {
            JOptionPane.showMessageDialog(this,
                    "Failed to open webcam. Check that a camera is connected.",
                    "Camera Error",
                    JOptionPane.ERROR_MESSAGE);
            return;
        }

        isCapturing = true;
        startStopButton.setText("■ Stop Capture");
        startStopButton.setBackground(new Color(198, 40, 40));
        updateStatus("Capturing...");

        // Timer 1: Refresh the video display (~30 FPS configurable)
        displayTimer = new Timer(AppConfig.DISPLAY_REFRESH_MS, e -> {
            BufferedImage frame = webcam.getBufferedImage();
            if (frame != null) {
                videoPanel.updateFrame(frame, new ArrayList<>(currentResults));
            }
        });
        displayTimer.start();

        // Timer 2: Send frames for recognition at configured interval
        recognitionTimer = new Timer(AppConfig.RECOGNITION_INTERVAL_MS, e -> {
            performRecognition();
        });
        recognitionTimer.start();
    }

    /**
     * Stop the webcam capture and timers.
     */
    private void stopCapture() {
        isCapturing = false;
        startStopButton.setText("▶ Start Capture");
        startStopButton.setBackground(new Color(46, 125, 50));

        if (displayTimer != null)
            displayTimer.stop();
        if (recognitionTimer != null)
            recognitionTimer.stop();

        webcam.stop();
        currentResults.clear();
        updateStatus("Stopped.");
    }

    // =========================================================================
    // Recognition Logic
    // =========================================================================

    /**
     * Capture current frame, send to backend for recognition,
     * and update the results. Runs in a background thread to
     * avoid freezing the GUI.
     */
    private void performRecognition() {
        // Get the current frame as base64
        String base64Frame = webcam.getBase64Frame();
        if (base64Frame == null)
            return;

        // Run API call in background thread
        new Thread(() -> {
            try {
                RecognizeResponse response = apiClient.recognize(base64Frame);

                // Update results (thread-safe)
                currentResults.clear();
                currentResults.addAll(response.results);

                // Update status on EDT
                SwingUtilities.invokeLater(() -> {
                    int faceCount = response.results.size();
                    long matchCount = response.results.stream()
                            .filter(r -> r.isCriminal())
                            .count();
                    long disguisedCount = response.results.stream()
                            .filter(r -> r.isDisguised)
                            .count();
                    updateStatus(String.format(
                            "Capturing... | Faces: %d | Criminals: %d | Disguised: %d",
                            faceCount, matchCount, disguisedCount));

                    // Add alerts only for confirmed criminal matches
                    for (RecognitionResult result : response.results) {
                        if (result.isCriminal()) {
                            // "Criminal: Yes" for a match, "No" would never appear here
                            String alertText = String.format(
                                    "⚠ %s: %s (%.1f%%) | Criminal: Yes | Disguise: %s",
                                    result.matchLevel,
                                    result.name,
                                    result.confidence * 100,
                                    result.isDisguised ? "YES" : "NO");
                            // Remove the placeholder message
                            if (alertsModel.getSize() == 1 &&
                                    alertsModel.get(0).startsWith("No alerts")) {
                                alertsModel.clear();
                            }
                            alertsModel.insertElementAt(alertText, 0);
                            // Keep alerts list manageable
                            if (alertsModel.getSize() > AppConfig.MAX_GUI_ALERTS) {
                                alertsModel.removeElementAt(alertsModel.getSize() - 1);
                            }
                        }
                    }
                });

            } catch (Exception ex) {
                SwingUtilities.invokeLater(() -> updateStatus("Error: " + ex.getMessage()));
            }
        }, "Recognition-Thread").start();
    }

    // =========================================================================
    // Add Criminal Dialog
    // =========================================================================

    /**
     * Show a dialog to add a new criminal to the database.
     * Captures the current frame and lets the user enter name + crime.
     */
    private void showAddCriminalDialog() {
        // Get current frame
        String base64Frame = webcam.getBase64Frame();
        if (base64Frame == null && isCapturing) {
            JOptionPane.showMessageDialog(this,
                    "No frame available. Make sure the camera is running.",
                    "No Frame", JOptionPane.WARNING_MESSAGE);
            return;
        }

        // Input dialog
        JPanel panel = new JPanel(new GridLayout(3, 2, 8, 8));
        panel.setBackground(new Color(60, 60, 60));
        JTextField nameField = new JTextField();
        JTextField crimeField = new JTextField();
        JLabel imageLabel = new JLabel(isCapturing ? "Current frame will be used" : "Start capture first");

        panel.add(createLabel("Name:"));
        panel.add(nameField);
        panel.add(createLabel("Crime:"));
        panel.add(crimeField);
        panel.add(createLabel("Image:"));
        panel.add(imageLabel);

        int result = JOptionPane.showConfirmDialog(
                this, panel, "Add Criminal to Database",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);

        if (result != JOptionPane.OK_OPTION)
            return;

        String name = nameField.getText().trim();
        String crime = crimeField.getText().trim();

        if (name.isEmpty() || crime.isEmpty()) {
            JOptionPane.showMessageDialog(this,
                    "Name and Crime fields are required.",
                    "Validation Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        // Re-capture frame for freshness
        String freshFrame = webcam.getBase64Frame();
        if (freshFrame == null)
            freshFrame = base64Frame;
        if (freshFrame == null) {
            JOptionPane.showMessageDialog(this,
                    "Cannot capture frame. Start the camera first.",
                    "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        // Send to backend in background
        final String frameToSend = freshFrame;
        new Thread(() -> {
            try {
                AddCriminalResponse resp = apiClient.addCriminal(name, crime, frameToSend);
                SwingUtilities.invokeLater(() -> {
                    JOptionPane.showMessageDialog(this,
                            resp.message,
                            resp.success ? "Success" : "Error",
                            resp.success ? JOptionPane.INFORMATION_MESSAGE : JOptionPane.ERROR_MESSAGE);
                });
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() -> {
                    JOptionPane.showMessageDialog(this,
                            "Failed to add criminal: " + ex.getMessage(),
                            "Error", JOptionPane.ERROR_MESSAGE);
                });
            }
        }, "AddCriminal-Thread").start();
    }

    private JLabel createLabel(String text) {
        JLabel label = new JLabel(text);
        label.setForeground(Color.WHITE);
        label.setFont(new Font("SansSerif", Font.PLAIN, 13));
        return label;
    }

    // =========================================================================
    // Alerts Refresh
    // =========================================================================

    /**
     * Fetch recent alerts from the backend and update the alerts list.
     */
    private void refreshAlerts() {
        new Thread(() -> {
            try {
                List<AlertInfo> alerts = apiClient.getAlerts(30);
                SwingUtilities.invokeLater(() -> {
                    alertsModel.clear();
                    if (alerts.isEmpty()) {
                        alertsModel.addElement("No alerts found.");
                    } else {
                        for (AlertInfo alert : alerts) {
                            alertsModel.addElement(alert.toString());
                        }
                    }
                });
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() -> {
                    alertsModel.clear();
                    alertsModel.addElement("Error fetching alerts: " + ex.getMessage());
                });
            }
        }, "RefreshAlerts-Thread").start();
    }

    // =========================================================================
    // Status Bar
    // =========================================================================

    private void updateStatus(String message) {
        statusLabel.setText("  Status: " + message);
    }

    // =========================================================================
    // Shutdown
    // =========================================================================

    private void shutdown() {
        System.out.println("[INFO] Shutting down...");
        if (displayTimer != null)
            displayTimer.stop();
        if (recognitionTimer != null)
            recognitionTimer.stop();
        webcam.stop();
    }

    // =========================================================================
    // Inner Class: Video Panel
    // =========================================================================

    /**
     * Custom JPanel that renders the webcam feed and overlays
     * bounding boxes, names, confidence scores, and disguise indicators.
     */
    private static class VideoPanel extends JPanel {

        private BufferedImage currentImage;
        private List<RecognitionResult> currentResults = new ArrayList<>();

        // Colors for drawing
        private static final Color COLOR_STRONG = new Color(220, 40, 40); // Red for strong matches
        private static final Color COLOR_POSSIBLE = new Color(255, 165, 0); // Orange for possible matches
        private static final Color COLOR_UNKNOWN = new Color(46, 200, 80); // Green for unknown
        private static final Color COLOR_DISGUISED = new Color(255, 200, 0); // Yellow for disguised indicator
        private static final Color COLOR_LABEL_BG = new Color(0, 0, 0, 200); // Semi-transparent black
        private static final Font FONT_NAME = new Font("SansSerif", Font.BOLD, 14);
        private static final Font FONT_CONFIDENCE = new Font("SansSerif", Font.PLAIN, 12);
        private static final Font FONT_DISGUISE = new Font("SansSerif", Font.BOLD, 11);

        public VideoPanel() {
            setBackground(Color.BLACK);
            setDoubleBuffered(true);
        }

        /**
         * Update the displayed frame and recognition results.
         *
         * @param image   New webcam frame.
         * @param results Recognition results to overlay.
         */
        public void updateFrame(BufferedImage image, List<RecognitionResult> results) {
            this.currentImage = image;
            this.currentResults = results != null ? results : new ArrayList<>();
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            if (currentImage == null) {
                // Draw placeholder text when no feed is available
                g2.setColor(new Color(100, 100, 100));
                g2.setFont(new Font("SansSerif", Font.BOLD, 18));
                String msg = "Camera feed not active. Click 'Start Capture' to begin.";
                FontMetrics fm = g2.getFontMetrics();
                int x = (getWidth() - fm.stringWidth(msg)) / 2;
                int y = getHeight() / 2;
                g2.drawString(msg, x, y);
                return;
            }

            // -----------------------------------------------------------------
            // Draw the webcam image, scaled to fit the panel while maintaining
            // the aspect ratio.
            // -----------------------------------------------------------------
            int panelW = getWidth();
            int panelH = getHeight();
            int imgW = currentImage.getWidth();
            int imgH = currentImage.getHeight();

            // Calculate scale to fit
            double scale = Math.min((double) panelW / imgW, (double) panelH / imgH);
            int scaledW = (int) (imgW * scale);
            int scaledH = (int) (imgH * scale);
            int offsetX = (panelW - scaledW) / 2;
            int offsetY = (panelH - scaledH) / 2;

            g2.drawImage(currentImage, offsetX, offsetY, scaledW, scaledH, null);

            // -----------------------------------------------------------------
            // Draw bounding boxes and labels for each recognition result
            // -----------------------------------------------------------------
            for (RecognitionResult result : currentResults) {
                BoundingBox box = result.boundingBox;

                // Scale bounding box coordinates to panel coordinates
                int bx = offsetX + (int) (box.x * scale);
                int by = offsetY + (int) (box.y * scale);
                int bw = (int) (box.width * scale);
                int bh = (int) (box.height * scale);

                // ─────────────────────────────────────────────────────────
                // Choose color based on match level (not just name)
                // ─────────────────────────────────────────────────────────
                String matchLevel = result.matchLevel != null ? result.matchLevel : "Unknown";
                Color boxColor;
                if ("Strong Match".equals(matchLevel)) {
                    boxColor = COLOR_STRONG; // Red — confirmed criminal
                } else if ("Possible Match".equals(matchLevel)) {
                    boxColor = COLOR_POSSIBLE; // Orange — needs review
                } else {
                    boxColor = COLOR_UNKNOWN; // Green — unknown person
                }

                // ─── Draw bounding box ───────────────────────────────────
                g2.setColor(boxColor);
                if (result.isDisguised) {
                    // Dashed border for disguised faces
                    g2.setStroke(new BasicStroke(3.0f, BasicStroke.CAP_BUTT,
                            BasicStroke.JOIN_MITER, 10.0f, new float[] { 8.0f, 4.0f }, 0.0f));
                } else {
                    g2.setStroke(new BasicStroke(2.5f));
                }
                g2.drawRect(bx, by, bw, bh);

                // ─── Build label text ────────────────────────────────────
                // Line 1: "Strong Match: Rahul (85%)"
                // Line 2: "Criminal: Yes"  or  "Criminal: No"
                // Line 3: "Disguise: YES"  or  "Disguise: NO"
                String confText     = String.format("%.1f%%", result.confidence * 100);
                String criminalTag  = result.isCriminal() ? "Yes" : "No";
                String disguiseTag  = result.isDisguised  ? "YES" : "NO";
                String line1 = String.format("%s: %s (%s)", matchLevel, result.name, confText);
                String line2 = String.format("Criminal: %s", criminalTag);
                String line3 = String.format("Disguise: %s", disguiseTag);

                g2.setFont(FONT_NAME);
                FontMetrics fmName = g2.getFontMetrics();
                g2.setFont(FONT_DISGUISE);
                FontMetrics fmSub = g2.getFontMetrics();

                int line1W  = fmName.stringWidth(line1);
                int line2W  = fmSub.stringWidth(line2);
                int line3W  = fmSub.stringWidth(line3);
                int labelW  = Math.max(line1W, Math.max(line2W, line3W)) + 14;
                int labelH  = fmName.getHeight() + fmSub.getHeight() * 2 + 10;

                // ─── Draw label background above the box ─────────────────
                int labelY = Math.max(by - labelH - 2, offsetY);
                g2.setColor(COLOR_LABEL_BG);
                g2.fillRoundRect(bx, labelY, labelW, labelH, 6, 6);

                // ─── Line 1: Match level + name + confidence ─────────────
                g2.setFont(FONT_NAME);
                g2.setColor(boxColor);
                g2.drawString(line1, bx + 6, labelY + fmName.getAscent() + 3);

                // ─── Line 2: Criminal Yes / No ───────────────────────────
                g2.setFont(FONT_DISGUISE);
                Color criminalColor = result.isCriminal()
                        ? new Color(255, 80, 80)   // red-ish for "Yes"
                        : new Color(100, 220, 100); // green-ish for "No"
                g2.setColor(criminalColor);
                int line2Y = labelY + fmName.getHeight() + fmSub.getAscent() + 4;
                g2.drawString(line2, bx + 6, line2Y);

                // ─── Line 3: Disguise status ─────────────────────────────
                g2.setColor(result.isDisguised ? COLOR_DISGUISED : Color.LIGHT_GRAY);
                int line3Y = line2Y + fmSub.getHeight() + 2;
                g2.drawString(line3, bx + 6, line3Y);
            }
        }
    }
}
