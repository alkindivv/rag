"use client";
import { useState } from "react";
import { ask } from "./actions";
import MessageList from "@/components/MessageList";
import ChatInput from "@/components/ChatInput";

export default function ChatPage() {
  const [messages, setMessages] = useState<any[]>([]);

  async function onSend(text: string) {
    const userMsg = { role: "user", content: text };
    setMessages((m) => [...m, userMsg]);
    const res = await ask(text);
    const assistantMsg = { role: "assistant", content: res.answer, sources: res.candidates };
    setMessages((m) => [...m, assistantMsg]);
  }

  return (
    <div className="mx-auto max-w-3xl p-4">
      <h1 className="text-2xl font-semibold mb-4">Legal RAG</h1>
      <MessageList messages={messages} />
      <ChatInput onSend={onSend} />
    </div>
  );
}
