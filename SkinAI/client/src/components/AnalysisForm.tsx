import { useState, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Sparkles, Upload, Image as ImageIcon, X } from "lucide-react";

interface AnalysisFormProps {
  onSubmit?: (data: { image: File }) => void;
  isLoading?: boolean;
}

export default function AnalysisForm({ onSubmit, isLoading }: AnalysisFormProps) {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedImage) {
      onSubmit?.({ image: selectedImage });
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <Card data-testid="card-analysis-form">
      <CardHeader>
        <CardTitle className="flex items-center gap-2" data-testid="text-form-title">
          <Sparkles className="h-5 w-5 text-primary" />
          제품 성분 분석
        </CardTitle>
        <CardDescription data-testid="text-form-description">
          제품 라벨 사진을 업로드하면 AI가 성분을 추출하고 안전성을 분석해드립니다
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
            data-testid="input-file"
          />

          {!previewUrl ? (
            <button
              type="button"
              onClick={handleUploadClick}
              className="w-full h-64 border-2 border-dashed rounded-lg hover-elevate flex flex-col items-center justify-center gap-4 text-muted-foreground transition-colors"
              data-testid="button-upload-area"
            >
              <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center">
                <ImageIcon className="h-8 w-8" />
              </div>
              <div className="text-center space-y-1">
                <p className="font-medium text-foreground">제품 라벨 사진 업로드</p>
                <p className="text-sm">클릭하여 이미지를 선택하세요</p>
              </div>
            </button>
          ) : (
            <div className="relative">
              <div className="w-full h-64 rounded-lg overflow-hidden bg-muted">
                <img
                  src={previewUrl}
                  alt="Selected product"
                  className="w-full h-full object-contain"
                  data-testid="img-preview"
                />
              </div>
              <Button
                type="button"
                variant="destructive"
                size="icon"
                className="absolute top-2 right-2"
                onClick={handleRemoveImage}
                data-testid="button-remove-image"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          )}

          <div className="flex gap-2">
            <Button
              type="submit"
              className="flex-1 gap-2"
              disabled={isLoading || !selectedImage}
              data-testid="button-analyze"
            >
              <Sparkles className="h-4 w-4" />
              {isLoading ? "분석 중..." : "분석 시작"}
            </Button>
            
            {selectedImage && (
              <Button
                type="button"
                variant="outline"
                onClick={handleUploadClick}
                disabled={isLoading}
                data-testid="button-change-image"
              >
                <Upload className="h-4 w-4 mr-2" />
                변경
              </Button>
            )}
          </div>

          <p className="text-xs text-muted-foreground text-center">
            제품 라벨의 성분표가 선명하게 보이는 사진을 업로드해주세요
          </p>
        </form>
      </CardContent>
    </Card>
  );
}
