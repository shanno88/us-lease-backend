from pathlib import Path
from typing import List
import fitz
from PIL import Image
import io


class PDFService:
    def __init__(self, dpi: int = 200):
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: Path) -> List[Path]:
        temp_image_paths = []

        try:
            doc = fitz.open(str(pdf_path))

            for page_num in range(len(doc)):
                page = doc[page_num]
                zoom = self.dpi / 72
                mat = fitz.Matrix(zoom, zoom)

                pix = page.get_pixmap(matrix=mat)

                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                temp_image_path = (
                    pdf_path.parent / f"{pdf_path.stem}_page_{page_num + 1}.png"
                )
                img.save(temp_image_path, "PNG")
                temp_image_paths.append(temp_image_path)

            doc.close()

            return temp_image_paths

        except Exception as e:
            for temp_path in temp_image_paths:
                if temp_path.exists():
                    temp_path.unlink()
            raise Exception(f"Failed to convert PDF to images: {str(e)}")

    def is_pdf(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def is_image(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}


_pdf_service = None


def get_pdf_service() -> PDFService:
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = PDFService()
    return _pdf_service
