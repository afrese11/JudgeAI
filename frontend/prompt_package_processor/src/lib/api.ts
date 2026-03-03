export interface JudgeCaseResponse {
  raw?: string;
  case_summary?: string | null;
  case_decision?: string | null;
  num_documents?: number;
  retrieval_k?: number;
  similar_cases?: unknown[];
  retrieval_error?: string | null;
  redact?: boolean;
  model?: string;
  error?: string;
}

interface SubmitJudgeCaseParams {
  files: File[];
  redact?: boolean;
  apiBase?: string;
}

const safeApiBase = (apiBase?: string) =>
  apiBase && apiBase.trim() ? apiBase.trim().replace(/\/+$/, "") : "";

export async function submitJudgeCase({
  files,
  redact = false,
  apiBase,
}: SubmitJudgeCaseParams): Promise<JudgeCaseResponse> {
  const endpoint = `${safeApiBase(apiBase)}/api/judge`;
  const form = new FormData();

  for (const file of files) {
    form.append("files", file, file.name);
  }
  form.append("redact", String(redact));

  const response = await fetch(endpoint, {
    method: "POST",
    body: form,
  });

  const payload: JudgeCaseResponse = await response
    .json()
    .catch(() => ({ error: "Server returned invalid JSON." }));

  if (!response.ok || payload.error) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }

  return {
    ...payload,
    similar_cases: Array.isArray(payload.similar_cases) ? payload.similar_cases : [],
  };
}
