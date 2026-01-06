import os
import subprocess
from datetime import datetime
import logging
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

class S3UploadCallback(TrainerCallback):
    """Callback to upload checkpoints to S3 after each save."""

    def __init__(self, bucket: str = "pcc-777-navy", output_dir: str = "output"):
        self.bucket = bucket
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y-%m-%d:%H-%M-%S")

    def on_save(self, args, state, control, **kwargs):
        """Called after checkpoint is saved."""
        checkpoint_dir = args.output_dir

        if not os.path.isdir(checkpoint_dir):
            logger.warning(f"[S3Upload] Output directory not found: {checkpoint_dir}")
            return

        s3_path = f"s3://{self.bucket}/{self.timestamp}/"
        logger.info("==========================================")
        logger.info("Uploading checkpoint to S3...")
        logger.info(f"Source: {checkpoint_dir}")
        logger.info(f"Destination: {s3_path}")
        logger.info("==========================================")

        try:
            result = subprocess.run(
                ["aws", "s3", "sync", checkpoint_dir, s3_path, "--quiet"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"S3 upload complete: {s3_path}")
            else:
                logger.warning(f"[WARN] S3 upload failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"[WARN] S3 upload error: {e}")
