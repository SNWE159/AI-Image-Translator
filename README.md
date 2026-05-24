# 🌍 TravelLens AI

## 🧠 AI-Powered Tourist Image Text Translator

TourTranslate AI is a Streamlit-based computer vision application designed to help international travelers understand foreign-language text in real-world environments.

Tourists often face difficulties understanding street signs, restaurant menus, transportation boards, and public notices in foreign countries. This application solves that problem instantly.

Users can simply upload an image of any signboard, and the system will detect, translate, and replace the text into English directly on the image.

---

## 🚀 Problem Statement

When traveling to a foreign country, tourists often encounter:

- Language barriers in public signs
- Difficulty understanding transport information
- Confusion in restaurants and shops
- Misinterpretation of important instructions

This can lead to confusion, delays, and poor travel experiences.

---

## 💡 Solution

TourTranslate AI allows users to:

📸 Upload any image containing text  
🔍 Automatically detect text using OCR  
🌐 Translate detected text into English  
🖼️ Replace original text with translated version  
📥 Download translated image instantly  

---

## 🏗️ System Architecture

Image Upload  
→ OCR Text Detection  
→ Language Identification  
→ Translation to English  
→ Text Removal (Image Processing)  
→ Text Rendering on Image  
→ Output Display  

---

## 🛠️ Tech Stack

- Streamlit (Frontend UI)
- EasyOCR (Text Detection)
- deep-translator (Translation Engine)
- OpenCV (Image Processing)
- Pillow (Image Rendering)
- NumPy (Data Processing)

---

## 🌐 Supported Languages

TourTranslate AI supports translation from:

- Spanish 🇪🇸  
- French 🇫🇷  
- German 🇩🇪  
- Italian 🇮🇹  
- Portuguese 🇵🇹  
- Dutch 🇳🇱  
- Japanese 🇯🇵  
- Korean 🇰🇷  
- Sinhala 🇱🇰  
- Arabic 🇸🇦  

➡️ All translated into English

---

## 📷 How It Works

1. Upload an image (street sign, menu, board, etc.)
2. System detects all text using OCR
3. Each text segment is translated to English
4. Original text is removed from image
5. Translated text is placed back into image
6. Final image is displayed and available for download

---

## 🎯 Use Cases

- ✈️ Tourism assistance
- 🏙️ Street navigation
- 🍽️ Restaurant menu translation
- 🚉 Transport signage understanding
- 🏨 Hotel & travel communication
- 📸 Travel photography enhancement

---

