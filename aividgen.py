import streamlit as st
import tempfile
import os
import subprocess
import torch
import logging
import gdown

class ImageAnimator:
    def __init__(self):
        self.setup_logging()
        self.setup_device()
        self.setup_ui()
        self.download_checkpoint()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
    
    def setup_ui(self):
        st.title("تحريك الصور باستخدام First Order Motion Model")
        st.write("قم برفع صورة ثابتة وفيديو مصدر الحركة لتحريك الصورة.")
        
        self.uploaded_image = st.file_uploader(
            "ارفع الصورة الثابتة",
            type=["png", "jpg", "jpeg"]
        )
        self.uploaded_video = st.file_uploader(
            "ارفع فيديو مصدر الحركة",
            type=["mp4", "avi", "mov"]
        )
    
    def download_checkpoint(self):
        # إنشاء مجلد checkpoints إذا لم يكن موجودًا
        os.makedirs("checkpoints", exist_ok=True)

        # رابط الملف على Google Drive
        file_id = "1sTRCQh2hTi3Z2oINRv3r_6NE7efCPB2e"
        output_path = "checkpoints/vox-cpk.pth.tar"

        # تنزيل الملف إذا لم يكن موجودًا
        if not os.path.exists(output_path):
            with st.spinner("جاري تنزيل الملف من Google Drive..."):
                gdown.download(id=file_id, output=output_path, quiet=False)
    
    def save_temp_file(self, uploaded_file, suffix):
        with tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.read())
            return temp_file.name
    
    def process_files(self):
        if not (self.uploaded_image and self.uploaded_video):
            st.info("يرجى رفع صورة وفيديو لبدء العملية.")
            return
        
        try:
            with st.spinner("جاري المعالجة..."):
                image_path = self.save_temp_file(self.uploaded_image, ".jpg")
                video_path = self.save_temp_file(self.uploaded_video, ".mp4")
                output_path = os.path.join("/tmp", "output.mp4")
                
                self.run_model(image_path, video_path, output_path)
                
                if os.path.exists(output_path):
                    st.success("تم إنشاء الفيديو بنجاح!")
                    st.video(output_path)
                
                # تنظيف الملفات
                os.remove(image_path)
                os.remove(video_path)
                
        except Exception as e:
            self.logger.error(f"حدث خطأ: {str(e)}")
            st.error(f"حدث خطأ أثناء المعالجة: {str(e)}")
    
    def run_model(self, image_path, video_path, output_path):
        command = [
            "python",
            "first_order_model/demo.py",
            "--config", "config/vox-256.yaml",
            "--checkpoint", "checkpoints/vox-cpk.pth.tar",
            "--source_image", image_path,
            "--driving_video", video_path,
            "--result_video", output_path,
            "--relative",
            "--adapt_scale"
        ]
        
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                raise Exception(process.stderr)
        except Exception as e:
            self.logger.error(f"حدث خطأ أثناء تشغيل النموذج: {str(e)}")
            raise

if __name__ == "__main__":
    animator = ImageAnimator()
    animator.process_files()