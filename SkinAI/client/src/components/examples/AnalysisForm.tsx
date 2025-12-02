import AnalysisForm from '../AnalysisForm';

export default function AnalysisFormExample() {
  return (
    <div className="max-w-2xl p-6">
      <AnalysisForm
        onSubmit={(data) => console.log('Image uploaded:', data.image.name)}
        isLoading={false}
      />
    </div>
  );
}
