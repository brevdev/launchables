import gradio as gr
import time
import json
import os
import fitz  # PyMuPDF
import base64
import io

from PIL import Image
from pathlib import Path
from gradio_pdf import PDF
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.util.file_processing.extract import extract_file_content
from langchain_core.documents import Document


def highlight_text(pdf_path, highlights, image_coordinates):
    """
    Highlights text and image bounding boxes in a PDF and creates a new PDF.

    Args:
        pdf_path (str): Path to the original PDF.
        highlights (dict): Dictionary containing page numbers as keys and bounding boxes for text as values.
        image_coordinates (dict): Dictionary containing page numbers as keys and bounding boxes for images as values.

    Returns:
        str: Path to the new highlighted PDF.
    """
    # Open the original PDF
    pdf_document = fitz.open(pdf_path)
    output_path = "highlighted_output.pdf"
    new_pdf_document = fitz.open()

    # Process each page in the original PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        new_page = new_pdf_document.new_page(width=page.rect.width, height=page.rect.height)
        
        # Draw the original page content
        new_page.show_pdf_page(page.rect, pdf_document, page_num)
        pdf_height = new_page.rect.height

        # Draw text highlights
        for bbox in highlights.get(page_num, []):
            # Highlight with a yellow rectangle
            new_page.draw_rect(bbox, color=(1, 1, 0), width=2)

        # Draw image bounding boxes
        if page_num in image_coordinates:
            for img_bbox in image_coordinates[page_num]:
                # Bounding box coordinates: [x0, y0, x1, y1]
                x0, y0, x1, y1 = img_bbox
                adjusted_y0 = pdf_height - y1  # Adjust for PDF coordinate system
                adjusted_y1 = pdf_height - y0
                # Draw a red rectangle around the image
                new_page.draw_rect([x0, adjusted_y0, x1, adjusted_y1], color=(1, 0, 0), width=2)

    # Save and close the new PDF
    new_pdf_document.save(output_path)
    new_pdf_document.close()
    pdf_document.close()
    
    return output_path

def get_text_coordinates(pdf_path, search_text):
    """
    Searches for multi-line text in a PDF and returns coordinates for each line.

    Args:
        pdf_path (str): Path to the PDF file.
        search_text (str): Text to search for in the PDF.

    Returns:
        dict: Dictionary containing page numbers as keys and lists of bounding boxes as values.
    """
    doc = fitz.open(pdf_path)
    text_coordinates = {}
    search_lines = search_text.splitlines()
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_coordinates[page_num] = []
    
        for search_line in search_lines:
            # Search for the exact match of the line
            bbox_list = page.search_for(search_line.strip())
            if bbox_list:
                text_coordinates[page_num].extend(bbox_list)
    
    doc.close()
    return text_coordinates

def base64_to_image(base64_str):
    """
    Converts a base64 string to a PIL Image.

    Args:
        base64_str (str): Base64-encoded string of an image.

    Returns:
        PIL.Image.Image: Decoded and resized PIL Image.
    """
    # Remove the base64 prefix if present
    if base64_str.startswith('data:image/'):
        base64_str = base64_str.split(',')[1]
    
    # Decode the base64 string
    image_data = base64.b64decode(base64_str)
    
    # Load the image from the decoded bytes
    image = Image.open(io.BytesIO(image_data))
    # Resize the image for display purposes
    image = image.resize((800, 600))  # Adjust size as needed
    return image

class PdfDemo:
    def __init__(self, client):
        self.client = client

    def process_pdf_file(self, file_path):
        start_time = time.time()

        file_content, file_type = extract_file_content(file_path) 

        job_spec = JobSpec(
            document_type=file_type,
            payload=file_content,
            source_id=file_path,
            source_name=file_path,
            extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
        )

        extract_task = ExtractTask(
            document_type=file_type,
            extract_text=True,
            extract_images=True,
            extract_tables=True,
        )
        
        job_spec.add_task(extract_task)
        job_id = self.client.add_job(job_spec)
        
        self.client.submit_job(job_id, "morpheus_task_queue")
        result = self.client.fetch_job_result(job_id, timeout=60)

        # Compute the time taken to process the PDF file
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_rounded = round(elapsed_time, 2) 

        return result, elapsed_time_rounded

    def extracted_multimodal_data(self, file_path):
        result, time_taken = self.process_pdf_file(file_path)
        table_chart_content = []
        image_base64_list = []
        image_content_location_by_page = {}
        full_text_content = ""

        for element in result[0]:
            if element['document_type'] == 'text':
                document = Document(element['metadata']['content'])
                full_text_content += document.page_content
            elif element['document_type'] == 'structured':
                table_content = Document(element['metadata']['table_metadata']['table_content'])
                table_chart_content.append(table_content)
            elif element['document_type'] == 'image':
                image_data = Document(element['metadata']['content']).page_content
                image_base64_list.append(image_data)
                page_number = element['metadata']['content_metadata']['hierarchy']['page']
                image_location = element['metadata']['image_metadata']['image_location']
                image_content_location_by_page.setdefault(page_number, []).append(image_location)

        json_data = json.dumps(result)

        return (
            full_text_content,
            table_chart_content,
            image_base64_list,
            image_content_location_by_page,
            json_data,
            time_taken,
        )
    
    def fetch_predefined_files_from_local(self):
        """
        Fetch predefined PDF files from a local folder.

        Args:
            folder_path (str): Path to the folder containing predefined PDF files.

        Returns:
            dict: A dictionary mapping file names to file paths.
        """
        predefined_files = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(script_dir, "pdf_examples")
        folder = Path(folder_path)  # Create a Path object from the folder path
        if folder.exists() and folder.is_dir():
            for pdf_file in folder.glob("*.pdf"):
                predefined_files[pdf_file.stem] = str(pdf_file)
        return predefined_files

    def run_analysis(self, file, highlight):
        (
            text,
            table_chart,
            image_base64_list,
            image_content_location,
            json_data,
            time_taken_for_processing,
        ) = self.extracted_multimodal_data(file)
        
        start_time = time.time()
        coordinates = get_text_coordinates(file, text)

        # Convert base64 strings to PIL images
        images = [base64_to_image(base64_str) for base64_str in image_base64_list]
        
        if highlight:
            highlighted_pdf_path = highlight_text(file, coordinates, image_content_location)
            end_time = time.time()
            elapsed_time_highlighting = end_time - start_time
            elapsed_time_rounded_highlighting = round(elapsed_time_highlighting, 2)
        else:
            highlighted_pdf_path = file
            elapsed_time_rounded_highlighting = 0
        
        return (
            highlighted_pdf_path,
            text,
            table_chart,
            images,
            json_data,
            f"**Time taken for extracting:** {time_taken_for_processing:.2f} (s)",
            f"**Time taken for highlighting:** {elapsed_time_rounded_highlighting:.2f} (s)"
        )

    def create_gradio_app(self):
        """
        Creates and launches the Gradio app for multimodal data extraction.
        """

        css = "styles.css"
        predefined_files = self.fetch_predefined_files_from_local()
        with gr.Blocks(css=css) as demo:
            gr.Markdown(
                """
                <div style="font-size: 36px; font-weight: bold; color: #76B900; text-align: center; margin-bottom: 20px;">
                    NVIDIA-INGEST: MULTI-MODAL DATA EXTRACTION
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Upload PDF File", elem_id="small-file-input", interactive=True
                    )
                    gr.Markdown(
                        """
                        <div style="font-size: 24px; text-align: center;">
                            Example pdfs
                        </div>
                        """
                    )

                    btn_actions = []
                    for file_name, file_path in predefined_files.items():
                        btn = gr.Button(file_name, variant="secondary")
                        btn_actions.append((btn, file_path))

                with gr.Column(scale=4):
                    bt_analysis = gr.Button("Run analysis")
                    with gr.Row():
                        time_taken_for_processing = gr.Markdown()
                    time_taken_for_highlighting = gr.Markdown()
                    bounding_box_toggle = gr.Checkbox(label="Highlight Extracted Data")
                    pdf_viewer = PDF(
                        label="Extract Multimodal Data", interactive=True, elem_id="pdf_viewer"
                    )

                with gr.Column(scale=4):
                    with gr.Tab("Content"):
                        text_content = gr.Textbox(label="Text", value="")
                        table_chart_content = gr.Textbox(label="Tables", value="")
                        image_gallery = gr.Gallery(
                            label="Charts & Images",
                            elem_id="gallery",
                            columns=[3],
                            rows=[1],
                            object_fit="contain",
                            height="auto",
                        )
                    with gr.Tab("Result"):
                        json_display = gr.JSON()

                # Link the file input to the PDF viewer
                file_input.change(
                    lambda file: file.name if file else None,
                    inputs=file_input,
                    outputs=pdf_viewer,
                )

                for btn, file_path in btn_actions:
                    btn.click(
                        lambda file_path=file_path: file_path,
                        inputs=None,
                        outputs=pdf_viewer,
                    )

                bt_analysis.click(
                    self.run_analysis,
                    inputs=[pdf_viewer, bounding_box_toggle],
                    outputs=[
                        pdf_viewer,
                        text_content,
                        table_chart_content,
                        image_gallery,
                        json_display,
                        time_taken_for_processing,
                        time_taken_for_highlighting,
                    ],
                )

        demo.launch(share=True)
