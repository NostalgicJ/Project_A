import AnalysisResult from "@/components/AnalysisResult";
import { ArrowRight, Camera, Search, Upload, X } from "lucide-react";
import { useState } from "react";

// ----------------------------------------------------------------
// 1. 데이터 타입 정의
// ----------------------------------------------------------------
interface AnalysisData {
  products: Array<{
    id: string;
    name: string;
    ingredients: string[];
  }>;
  analysis: {
    score: number;
    status: "SAFE" | "CAUTION" | "UNKNOWN";
    message: string;
    problematic_ingredients?: string[]; 
  };
}

type Ingredient = {
  name: string;
  safetyLevel: "safe" | "caution"; 
};

export default function Analyze() {
  // ----------------------------------------------------------------
  // 2. 상태 관리
  // ----------------------------------------------------------------
  const [showResult, setShowResult] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [inputMode, setInputMode] = useState<'image' | 'text'>('image');
  
  // 텍스트 입력 상태
  const [productName1, setProductName1] = useState("");
  const [productName2, setProductName2] = useState("");

  // 이미지 파일 상태 (제품 2개)
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);

  // 분석 결과 데이터
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);

  // ----------------------------------------------------------------
  // 3. 텍스트 분석 핸들러
  // ----------------------------------------------------------------
  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!productName1 || !productName2) {
      alert("두 개의 제품명을 모두 입력해주세요.");
      return;
    }
    startAnalysis();

    try {
      const response = await fetch('/api/analyze/text', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ 
              product1_name: productName1, 
              product2_name: productName2 
          })
      });

      const data = await response.json();

      if (response.ok) {
        setAnalysisData(data);
        setShowResult(true);
      } else {
        alert(data.error || "제품을 찾을 수 없습니다.");
      }
    } catch (error) {
      alert("서버 통신 오류가 발생했습니다.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ----------------------------------------------------------------
  // 4. 이미지 분석 핸들러
  // ----------------------------------------------------------------
  const handleImageAnalyze = async () => {
    if (!image1 || !image2) {
      alert("두 제품의 사진을 모두 등록해주세요.");
      return;
    }
    
    startAnalysis();
    
    const formData = new FormData();
    formData.append("image1", image1);
    formData.append("image2", image2);

    try {
      // OCR 분석 요청
      const response = await fetch('/api/analyze/image', {
          method: 'POST',
          body: formData, 
      });
      
      const data = await response.json();

      if (response.ok) {
        console.log("✅ 이미지 분석 성공:", data);
        setAnalysisData(data);
        setShowResult(true);
      } else {
        alert("이미지 분석 실패: " + (data.error || "알 수 없는 오류"));
      }
    } catch (error) {
      console.error(error);
      alert("서버 통신 오류가 발생했습니다.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const startAnalysis = () => {
    setIsAnalyzing(true);
    setAnalysisData(null);
  };

  const resetForm = () => {
    setShowResult(false);
    setAnalysisData(null);
    setProductName1("");
    setProductName2("");
    setImage1(null);
    setImage2(null);
  };

  // ----------------------------------------------------------------
  // 5. 데이터 가공 (화면 표시용)
  // ----------------------------------------------------------------
  const getDisplayData = () => {
    if (!analysisData) return null;

    const p1Ings = analysisData.products[0]?.ingredients || [];
    const p2Ings = analysisData.products[1]?.ingredients || [];
    const allIngs = Array.from(new Set([...p1Ings, ...p2Ings]));
    
    const culprits = analysisData.analysis.problematic_ingredients || [];

    const formattedIngredients: Ingredient[] = allIngs.map(name => ({
      name: name,
      safetyLevel: culprits.includes(name) ? "caution" : "safe" 
    }));

    const isSafe = analysisData.analysis.status === 'SAFE';
    
    return {
      productName: `${analysisData.products[0].name} & ${analysisData.products[1].name}`,
      brand: isSafe ? "✅ 안전한 조합" : "🚨 주의 필요",
      score: analysisData.analysis.score,
      ingredients: formattedIngredients,
      summary: analysisData.analysis.message,
      warnings: !isSafe 
        ? ["이 조합에서 주의가 필요한 성분(들)을 발견했습니다.", "피부 타입에 따라 자극이 있을 수 있습니다."] 
        : [],
      recommendations: isSafe ? ["안심하고 사용하셔도 좋습니다."] : []
    };
  };

  const displayData = getDisplayData();

  // UI 컴포넌트 (이미지 미리보기)
  const renderImagePreview = (file: File | null, setFile: (f: File | null) => void, label: string) => (
    <div className="border-2 border-dashed border-gray-200 rounded-xl p-3 hover:bg-gray-50 transition-colors relative group h-32 flex items-center justify-center bg-gray-50/30">
      {file ? (
        <div className="relative h-full w-full">
           <img src={URL.createObjectURL(file)} alt="preview" className="h-full w-full object-contain rounded-lg" />
           <button 
             onClick={(e) => { e.preventDefault(); setFile(null); }}
             className="absolute -top-2 -right-2 bg-white rounded-full p-1 shadow-md hover:bg-gray-100 border"
           >
             <X className="w-4 h-4 text-gray-500" />
           </button>
           <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-xs p-1 rounded-b-lg truncate px-2 text-center">
             {file.name}
           </div>
        </div>
      ) : (
        <label className="cursor-pointer flex flex-col items-center justify-center h-full w-full gap-2">
          <input type="file" accept="image/*" className="hidden" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <Upload className="w-8 h-8 text-gray-400" />
          <span className="text-sm text-gray-500 font-medium">{label} 업로드</span>
        </label>
      )}
    </div>
  );

  return (
    <div className="min-h-screen py-12 px-6 bg-gray-50/50">
      <div className="container max-w-6xl mx-auto">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold mb-4 text-gray-900" data-testid="text-analyze-title">
            성분 궁합 분석
          </h1>
          <p className="text-muted-foreground text-lg">
            사용하고 있는 두 제품의 성분 궁합을 AI가 분석해드립니다.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-10 items-start">
          
          {/* [왼쪽] 입력 영역 */}
          <div className="lg:sticky lg:top-24 space-y-6">
            
            {/* 탭 버튼 */}
            <div className="bg-white p-1 rounded-xl border shadow-sm flex">
              <button onClick={() => setInputMode('image')} className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${inputMode === 'image' ? 'bg-primary/10 text-primary ring-2 ring-primary/20 font-bold' : 'text-gray-500 hover:bg-gray-100'}`}>
                <Camera className="w-4 h-4 inline mr-2"/> 사진으로 분석
              </button>
              <button onClick={() => setInputMode('text')} className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${inputMode === 'text' ? 'bg-primary/10 text-primary ring-2 ring-primary/20 font-bold' : 'text-gray-500 hover:bg-gray-100'}`}>
                <Search className="w-4 h-4 inline mr-2"/> 검색
              </button>
            </div>

            <div className="bg-white p-6 rounded-2xl border shadow-sm">
              {inputMode === 'image' ? (
                // 1. 사진 2장 업로드 UI
                <div className="space-y-6">
                  <div>
                    <h3 className="font-semibold text-lg mb-2">제품 사진 촬영</h3>
                    <p className="text-sm text-gray-500 mb-4">두 제품의 성분표가 잘 보이도록 각각 찍어주세요.</p>
                    <div className="grid grid-cols-2 gap-4">
                      {renderImagePreview(image1, setImage1, "첫 번째 제품")}
                      {renderImagePreview(image2, setImage2, "두 번째 제품")}
                    </div>
                  </div>
                  {/* 버튼 색상: bg-primary */}
                  <button onClick={handleImageAnalyze} disabled={isAnalyzing} className="w-full bg-primary text-white hover:bg-primary/90 disabled:opacity-50 py-4 rounded-xl font-medium flex items-center justify-center gap-2 transition-all shadow-md hover:shadow-lg">
                    {isAnalyzing ? "분석 중... (OCR 진행)" : <>사진으로 분석하기 <ArrowRight className="w-4 h-4" /></>}
                  </button>
                </div>
              ) : (
                // 2. 텍스트 입력 폼
                <form onSubmit={handleTextSubmit} className="space-y-6">
                  <div>
                    <h3 className="font-semibold text-lg mb-2">제품명 직접 입력</h3>
                    <p className="text-sm text-gray-500 mb-4">정확한 제품명을 입력해야 분석이 가능합니다.</p>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <label className="text-xs font-bold text-gray-500 uppercase tracking-wider">Product 01</label>
                        <input type="text" placeholder="예: 닥터지 레드 블레미쉬 크림" value={productName1} onChange={(e)=>setProductName1(e.target.value)} className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none transition-all" />
                      </div>
                      <div className="space-y-2">
                        <label className="text-xs font-bold text-gray-500 uppercase tracking-wider">Product 02</label>
                        <input type="text" placeholder="예: 이니스프리 레티놀 시카 앰플" value={productName2} onChange={(e)=>setProductName2(e.target.value)} className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none transition-all" />
                      </div>
                    </div>
                  </div>
                  {/* 버튼 색상: bg-primary */}
                  <button type="submit" disabled={isAnalyzing} className="w-full bg-primary text-white hover:bg-primary/90 disabled:opacity-50 py-4 rounded-xl font-medium flex items-center justify-center gap-2 transition-all shadow-md hover:shadow-lg">
                    {isAnalyzing ? "분석 중..." : <>궁합 분석하기 <ArrowRight className="w-4 h-4" /></>}
                  </button>
                </form>
              )}
            </div>
          </div>

          {/* [오른쪽] 결과 영역 */}
          <div>
            {showResult && displayData ? (
              <AnalysisResult
                productName={displayData.productName}
                brand={displayData.brand}
                safetyScore={displayData.score}
                ingredients={displayData.ingredients}
                summary={displayData.summary}
                warnings={displayData.warnings}
                recommendations={displayData.recommendations}
                onAddToCollection={() => alert("내 화장대에 저장되었습니다!")}
                onCancel={resetForm}
              />
            ) : (
              <div className="h-[500px] flex flex-col items-center justify-center bg-white rounded-2xl border-2 border-dashed border-gray-200 text-center p-8">
                <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mb-4">
                  <Search className="w-8 h-8 text-gray-300" />
                </div>
                <h3 className="font-semibold text-lg text-gray-900 mb-2">분석 대기 중</h3>
                <p className="text-muted-foreground max-w-xs" data-testid="text-empty-result">
                  왼쪽에서 제품을 등록하고<br/>분석을 시작하세요
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
