from flask import Flask, request, jsonify, send_file
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 업로드 허용 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 임시 파일 저장 디렉토리
UPLOAD_DIR = "./uploads"
RESULT_DIR = "./results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# OCR 모델 초기화 (서버 시작시 한 번만)
print("=" * 50)
print("PaddleOCR 모델 로딩 중...")
ocr = PaddleOCR(
    lang="korean",
    det_db_thresh=0.5,
    det_db_box_thresh=0.4,
    use_angle_cls=True
)
print("PaddleOCR 모델 로딩 완료!")
print("=" * 50)

def allowed_file(filename):
    """파일 확장자 검증"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """API 상태 확인"""
    return jsonify({
        "message": "Korean OCR API is running!",
        "endpoints": {
            "/ocr": "POST - 이미지에서 텍스트 추출",
            "/ocr-with-image": "POST - 텍스트 추출 + 결과 이미지 생성",
            "/download/<filename>": "GET - 결과 이미지 다운로드"
        }
    })

@app.route('/ocr', methods=['POST'])
def extract_text():
    """
    이미지를 업로드하면 OCR로 텍스트를 추출합니다.
    """
    try:
        # 파일 검증
        if 'file' not in request.files:
            return jsonify({"error": "파일이 없습니다."}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "파일이 선택되지 않았습니다."}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "지원하지 않는 파일 형식입니다."}), 400
        
        # 임시 파일로 저장
        temp_filename = f"{uuid.uuid4()}.jpg"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        image = Image.open(file.stream).convert('RGB')
        image.save(temp_path)
        
        # OCR 실행
        result = ocr.predict(temp_path)
        data = result[0]
        
        # 결과 추출
        texts = data['rec_texts']
        scores = data['rec_scores']
        boxes = data['rec_polys']
        
        # 결과 포맷팅
        ocr_results = []
        for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
            ocr_results.append({
                "index": i + 1,
                "text": text,
                "confidence": round(float(score), 2),
                "box": [[int(point[0]), int(point[1])] for point in box]
            })
        
        # 임시 파일 삭제
        os.remove(temp_path)
        
        return jsonify({
            "success": True,
            "total_texts": len(texts),
            "results": ocr_results
        })
        
    except Exception as e:
        return jsonify({"error": f"OCR 처리 중 오류 발생: {str(e)}"}), 500

@app.route('/ocr-with-image', methods=['POST'])
def extract_text_with_image():
    """
    이미지를 업로드하면 OCR로 텍스트를 추출하고, 
    텍스트와 박스가 표시된 결과 이미지를 생성합니다.
    """
    try:
        # 파일 검증
        if 'file' not in request.files:
            return jsonify({"error": "파일이 없습니다."}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "파일이 선택되지 않았습니다."}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "지원하지 않는 파일 형식입니다."}), 400
        
        # 임시 파일로 저장
        temp_filename = f"{uuid.uuid4()}.jpg"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        image = Image.open(file.stream).convert('RGB')
        image.save(temp_path)
        
        # OCR 실행
        result = ocr.predict(temp_path)
        data = result[0]
        
        # 결과 추출
        texts = data['rec_texts']
        scores = data['rec_scores']
        boxes = data['rec_polys']
        
        # 이미지에 결과 그리기
        draw = ImageDraw.Draw(image)
        
        # 폰트 설정 (시스템 폰트 또는 기본 폰트)
        try:
            # Linux/Railway 환경을 고려한 폰트 경로
            font_paths = [
                '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                'C:/Windows/Fonts/malgun.ttf'
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 25)
                    break
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 박스와 텍스트 그리기
        for text, score, box in zip(texts, scores, boxes):
            box_points = [(int(point[0]), int(point[1])) for point in box]
            draw.polygon(box_points, outline='green', width=2)
            draw.text((box_points[0][0], box_points[0][1] - 25), 
                     f'{text} ({score:.2f})', 
                     fill='red', font=font)
        
        # 결과 이미지 저장
        result_filename = f"result_{uuid.uuid4()}.jpg"
        result_path = os.path.join(RESULT_DIR, result_filename)
        image.save(result_path)
        
        # 임시 파일 삭제
        os.remove(temp_path)
        
        # 결과 포맷팅
        ocr_results = []
        for i, (text, score) in enumerate(zip(texts, scores)):
            ocr_results.append({
                "index": i + 1,
                "text": text,
                "confidence": round(float(score), 2)
            })
        
        return jsonify({
            "success": True,
            "total_texts": len(texts),
            "results": ocr_results,
            "result_image_url": f"/download/{result_filename}"
        })
        
    except Exception as e:
        return jsonify({"error": f"OCR 처리 중 오류 발생: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_result(filename):
    """결과 이미지 다운로드"""
    file_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "파일을 찾을 수 없습니다."}), 404
    return send_file(file_path, mimetype='image/jpeg')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    app.run(host="0.0.0.0", port=port, debug=False)
