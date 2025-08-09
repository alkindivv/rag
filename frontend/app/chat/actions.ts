"use server";

export async function ask(query: string) {
  const r = await fetch(process.env.BACKEND_URL + "/ask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ query })
  });
  if (!r.ok) throw new Error("API error");
  return r.json();
}
