import MessageBubble from "./MessageBubble";

export default function MessageList({ messages }: { messages: any[] }) {
  return (
    <div className="space-y-4">
      {messages.map((m, i) => (
        <MessageBubble key={i} message={m} />
      ))}
    </div>
  );
}
