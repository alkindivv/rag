export default function SourceCard({ source }: { source: any }) {
  return (
    <div className="border p-2 text-sm rounded">
      <div className="font-semibold">{source.citation}</div>
      <div>{source.text}</div>
    </div>
  );
}
