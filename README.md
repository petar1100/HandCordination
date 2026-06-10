# 🎮 Minecraft Hand Gesture Controller

Control Minecraft using your hands and a webcam! This Python script leverages **MediaPipe** and **OpenCV** to track your hand landmarks, mapping real-time hand movements and gestures into direct keyboard and mouse controls. It features custom hardware-level mouse injection to bypass camera locking engines typical in 3D games like Minecraft.

---

## ✨ Features

* **Dual-Hand Layout:** Separate tracking for player movement/actions (Left hand) and looking around (Right hand).
* **DirectInput Bypass:** Uses hardware-level Win32 API calls (`mouse_event`) to allow 360° fluid camera look inside Minecraft.
* **Jitter Reduction:** Features a built-in dead zone filter and linear smoothing configurations for stable aiming.
* **Live HUD:** On-screen OpenCV display visualizing your current gesture, directional tracking vectors, and tracking status.

---

## 🖐️ Gesture Mapping Guide

| Hand | Gesture | Action mapped in Game |
| :--- | :--- | :--- |
| **Right Hand** | ☝️ Point Index Finger | **Mouse Look** (Move camera by moving finger) |
| **Left Hand** | ☝️ Index Pointing UP | **Move Forward** (`W`) |
| **Left Hand** | ✌️ Index + Middle UP | **Move Backward** (`S`) |
| **Left Hand** | 👉 Index Pointing RIGHT | **Strafe Right** (`D`) |
| **Left Hand** | 👈 Index + Middle RIGHT | **Strafe Left** (`A`) |
| **Left Hand** | 👍 Thumbs UP | **Jump** (`Spacebar`) |
| **Left Hand** | 🤙 Thumb + Pinky UP | **Sneak** (`Shift`) |
| **Left Hand** | 🤘 Index + Pinky UP | **Break / Attack** (`Left Click Hold`) |
| **Left Hand** | 🖕 Pinky Only UP | **Build / Use Item** (`Right Click Hold`) |
| **Left Hand** | 🖐️ Open Hand | **Toggle Inventory** (`E` - Cooldown enabled) |
| **Left Hand** | 👊 Closed Fist / Rest | **Stop all movement / inputs** |

---

## 🚀 Getting Started

### 1. Prerequisites
This script requires **Windows** (due to the Win32 mouse event injection) and a working **webcam**.

### 2. Installation
Clone this repository or save the script file, then install the required dependencies using pip:

```bash
pip install opencv-python mediapipe pynput pyautogui
