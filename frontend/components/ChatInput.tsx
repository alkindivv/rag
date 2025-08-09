"use client";
import { useState } from "react";

export default function ChatInput({ onSend }: { onSend: (v: string) => Promise<void> }) {
  const [value, setValue] = useState("");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!value.trim()) return;
    await onSend(value);
    setValue("");
  }

  return (
    <form onSubmit={submit} className="flex gap-2 mt-4">
      <input
        className="flex-1 border p-2"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <button className="px-4 py-2 bg-blue-500 text-white" type="submit">
        Send
      </button>
    </form>
  );
}
