import SourceCard from "./SourceCard";

export default function MessageBubble({ message }: { message: any }) {
  return (
    <div className={message.role === "user" ? "text-right" : ""}>
      <div className="p-2 bg-gray-100 rounded">{message.content}</div>
      {message.sources && (
        <div className="mt-2 space-y-1">
          {message.sources.map((s: any, i: number) => (
            <SourceCard key={i} source={s} />
          ))}
        </div>
      )}
    </div>
  );
}
