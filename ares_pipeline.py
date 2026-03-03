import os
import torch
import segmentation_models_pytorch as smp
import numpy as np
from telegram import Bot
import asyncio

class AresPipeline:
    def __init__(self, telegram_token=None, telegram_chat_id=None):
        """
        Initializes the AresPipeline.
        
        Args:
            telegram_token (str): Bot token for Telegram notifications.
            telegram_chat_id (str): Chat ID for Telegram notifications.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Ares Pipeline on {self.device}...")
        
        # Load U-Net with ResNet-34 encoder
        # Using 'imagenet' pre-trained weights for the encoder
        self.model = smp.Unet(
            encoder_name="resnet34",        
            encoder_weights="imagenet",     
            in_channels=3,                  
            classes=1,                      
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        print("Model loaded successfully.")

    def physics_validation(self, mask, hand_data):
        """
        Placeholder for physics-based validation using HAND and JRC data.
        
        Args:
            mask (np.array): Binary flood mask from the U-Net model.
            hand_data (np.array): HAND (Height Above Nearest Drainage) data.
            
        Returns:
            np.array: Validated flood mask.
        """
        # TODO: Implement cross-referencing with HAND and JRC Global Surface Water Mask
        # Logic: IF (Detection == True AND HAND value is low AND Not Permanent Water) -> Confirm Flood
        
        print("Running physics validation (Placeholder)...")
        # For now, just return the mask as is, or apply a dummy filter
        validated_mask = mask # Pass-through for now
        return validated_mask

    def send_telegram_alert(self, message):
        """
        Sends a critical alert via Telegram.
        
        Args:
            message (str): The message content to send.
        """
        if not self.telegram_token or not self.telegram_chat_id:
            print("Telegram credentials not provided. Skipping alert.")
            return

        print(f"Sending Telegram Alert: {message}")
        
        async def _send():
            bot = Bot(token=self.telegram_token)
            await bot.send_message(chat_id=self.telegram_chat_id, text=message)

        try:
            asyncio.run(_send())
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")

    def run(self, image, hand_data=None):
        """
        Executes the full detection pipeline.
        
        Args:
            image (np.array): Input 3-channel satellite image (H, W, 3).
            hand_data (np.array): Input HAND data for validation.
            
        Returns:
            dict: Result containing the mask and alert status.
        """
        # Preprocess image
        # smp.Unet expects (Batch, Channel, Height, Width)
        # Normalize and convert to tensor
        input_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            # Sigmoid to get probability, then 0.5 threshold
            prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            binary_mask = (prob_mask > 0.5).astype(np.uint8)
            
        # Physics Validation
        validated_mask = self.physics_validation(binary_mask, hand_data)
        
        # Check for flood detection
        flood_detected = np.any(validated_mask == 1)
        
        if flood_detected:
            # Simulate alert generation (In reality, we'd calculate coordinates, etc.)
            alert_msg = (
                "🚨 *ARES FLOOD ALERT* 🚨\n"
                "- Location: [Coordinates/District]\n"
                "- Confidence Score: [U-Net %]\n"
                "- Physics Validation: [HAND Confirmed]\n"
                "- Satellite Source: Sentinel-2 (10m Resolution)\n"
                "Flood detected in the processed region."
            )
            self.send_telegram_alert(alert_msg)
            
        return {
            "raw_mask": binary_mask,
            "validated_mask": validated_mask,
            "flood_detected": flood_detected
        }

if __name__ == "__main__":
    # Example usage
    pipeline = AresPipeline()
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    pipeline.run(dummy_image)
