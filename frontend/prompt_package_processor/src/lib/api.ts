export interface JudgeCaseResponse {
  raw?: string;
  case_summary?: string | null;
  case_decision?: string | null;
  oral_argument_raw?: string | null;
  oral_argument_prediction?: string | null;
  oral_argument_summary?: string | null;
  oral_argument_error?: string | null;
  num_documents?: number;
  retrieval_k?: number;
  similar_cases?: unknown[];
  retrieval_error?: string | null;
  redact?: boolean;
  model?: string;
  error?: string;
  detail?: string | { msg?: string }[] | null;
}

interface SubmitJudgeCaseParams {
  files: File[];
  redact?: boolean;
  apiBase?: string;
  accessToken?: string;
}

const safeApiBase = (apiBase?: string) =>
  apiBase && apiBase.trim() ? apiBase.trim().replace(/\/+$/, "") : "";

export async function submitJudgeCase({
  files,
  redact = false,
  apiBase,
  accessToken,
}: SubmitJudgeCaseParams): Promise<JudgeCaseResponse> {
  const endpoint = `${safeApiBase(apiBase)}/api/judge`;
  const form = new FormData();

  for (const file of files) {
    form.append("files", file, file.name);
  }
  form.append("redact", String(redact));

  const response = await fetch(endpoint, {
    method: "POST",
    headers: accessToken
      ? {
          Authorization: `Bearer ${accessToken}`,
        }
      : undefined,
    body: form,
  });

  const payload: JudgeCaseResponse = await response
    .json()
    .catch(() => ({ error: "Server returned invalid JSON." }));

  const errorDetail = (() => {
    if (payload.error) return payload.error;
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
      const messages = payload.detail
        .map((item) => (item && typeof item.msg === "string" ? item.msg : ""))
        .filter(Boolean);
      if (messages.length > 0) return messages.join("; ");
    }
    return null;
  })();

  if (!response.ok || errorDetail) {
    throw new Error(errorDetail || `Request failed (${response.status})`);
  }

  return {
    ...payload,
    similar_cases: Array.isArray(payload.similar_cases) ? payload.similar_cases : [],
  };
}
