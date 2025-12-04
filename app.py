from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import os
import json
import uuid
from datetime import datetime

app = Flask(__name__)

# 디렉토리 생성
UPLOAD_DIR = "./uploads"
RESULT_DIR = "./ocr_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# OCR 모델 초기화
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

@app.route('/')
def home():
    """API 상태 확인"""
    return jsonify({
        "message": "Korean OCR API is running!",
        "status": "healthy",
        "endpoints": {
            "/upload": "POST - Unity에서 이미지 업로드 (파일 직접 전송)",
            "/ocr/process": "POST - 서버 이미지 경로로 OCR",
            "/ocr/batch": "POST - 여러 이미지 배치 처리"
        }
    })

@app.route('/health')
def health():
    """헬스체크"""
    return jsonify({"status": "healthy"}), 200

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Unity에서 이미지 파일을 직접 업로드받아 OCR 처리
    Unity가 기대하는 응답 형식: {"result": "텍스트", "confidence": 0.95}
    """
    try:
        # 파일 검증
        if 'file' not in request.files:
            return jsonify({"error": "파일이 없습니다."}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "파일이 선택되지 않았습니다."}), 400
        
        # 임시 파일로 저장
        temp_filename = f"{uuid.uuid4()}.jpg"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        # 이미지 저장
        image = Image.open(file.stream).convert('RGB')
        image.save(temp_path)
        
        # OCR 실행
        print(f"OCR 처리 시작: {temp_filename}")
        result = ocr.predict(temp_path)
        ocr_data = result[0]
        
        # 결과 추출
        texts = ocr_data['rec_texts']
        scores = ocr_data['rec_scores']
        
        # 모든 텍스트를 하나로 합치기
        full_text = " ".join(texts)
        
        # 평균 신뢰도 계산
        avg_confidence = sum(scores) / len(scores) if scores else 0.0
        
        # 임시 파일 삭제
        os.remove(temp_path)
        
        print(f"OCR 처리 완료: {full_text} (신뢰도: {avg_confidence:.2f})")
        
        # Unity가 기대하는 형식으로 응답
        return jsonify({
            "result": full_text,
            "confidence": round(float(avg_confidence), 2)
        })
        
    except Exception as e:
        print(f"OCR 오류: {str(e)}")
        return jsonify({
            "error": f"OCR 처리 중 오류 발생: {str(e)}",
            "result": "",
            "confidence": 0.0
        }), 500

@app.route('/ocr/process', methods=['POST'])
def process_server_image():
    """
    서버에 저장된 이미지 경로를 받아서 OCR 처리 후 결과를 서버에 저장
    
    요청 JSON:
    {
        "image_path": "/path/to/image.jpg",
        "save_result": true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image_path' not in data:
            return jsonify({"error": "image_path가 필요합니다."}), 400
        
        image_path = data['image_path']
        save_result = data.get('save_result', True)
        
        # 파일 존재 확인
        if not os.path.exists(image_path):
            return jsonify({"error": f"이미지를 찾을 수 없습니다: {image_path}"}), 404
        
        # OCR 실행
        print(f"OCR 처리 시작: {image_path}")
        result = ocr.predict(image_path)
        ocr_data = result[0]
        
        # 결과 추출
        texts = ocr_data['rec_texts']
        scores = ocr_data['rec_scores']
        boxes = ocr_data['rec_polys']
        
        # 결과 포맷팅
        ocr_results = []
        for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
            ocr_results.append({
                "index": i + 1,
                "text": text,
                "confidence": round(float(score), 2),
                "box": [[int(point[0]), int(point[1])] for point in box]
            })
        
        response_data = {
            "success": True,
            "image_path": image_path,
            "total_texts": len(texts),
            "results": ocr_results,
            "processed_at": datetime.now().isoformat()
        }
        
        # 결과를 서버에 JSON 파일로 저장
        if save_result:
            base_name = os.path.basename(image_path)
            result_filename = f"{os.path.splitext(base_name)[0]}_ocr_result.json"
            result_path = os.path.join(RESULT_DIR, result_filename)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            
            response_data["result_saved_to"] = result_path
            print(f"OCR 결과 저장: {result_path}")
        
        print(f"OCR 처리 완료: {len(texts)}개의 텍스트 추출")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"OCR 오류: {str(e)}")
        return jsonify({"error": f"OCR 처리 중 오류 발생: {str(e)}"}), 500

@app.route('/ocr/batch', methods=['POST'])
def process_batch():
    """
    여러 이미지를 한 번에 처리
    
    요청 JSON:
    {
        "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
        "save_result": true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image_paths' not in data:
            return jsonify({"error": "image_paths가 필요합니다."}), 400
        
        image_paths = data['image_paths']
        save_result = data.get('save_result', True)
        
        batch_results = []
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                batch_results.append({
                    "image_path": image_path,
                    "success": False,
                    "error": "파일을 찾을 수 없습니다."
                })
                continue
            
            try:
                # OCR 실행
                result = ocr.predict(image_path)
                ocr_data = result[0]
                
                texts = ocr_data['rec_texts']
                scores = ocr_data['rec_scores']
                boxes = ocr_data['rec_polys']
                
                ocr_results = []
                for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
                    ocr_results.append({
                        "index": i + 1,
                        "text": text,
                        "confidence": round(float(score), 2),
                        "box": [[int(point[0]), int(point[1])] for point in box]
                    })
                
                result_data = {
                    "success": True,
                    "image_path": image_path,
                    "total_texts": len(texts),
                    "results": ocr_results
                }
                
                # 결과 저장
                if save_result:
                    base_name = os.path.basename(image_path)
                    result_filename = f"{os.path.splitext(base_name)[0]}_ocr_result.json"
                    result_path = os.path.join(RESULT_DIR, result_filename)
                    
                    with open(result_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=2)
                    
                    result_data["result_saved_to"] = result_path
                
                batch_results.append(result_data)
                
            except Exception as e:
                batch_results.append({
                    "image_path": image_path,
                    "success": False,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "total_processed": len(batch_results),
            "results": batch_results
        })
        
    except Exception as e:
        return jsonify({"error": f"배치 처리 중 오류: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"서버 시작: 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
