import { z } from 'zod';
import { context } from '@opentelemetry/api';
import { suppressTracing } from '@opentelemetry/core';

/** Renew only the CLI task's selected host/project. No refresh token crosses this boundary. */
export function taskCredentialRefresh(options: { token: string; projectId: string; baseUrl: string }) {
  const endpoint = process.env.NEATLOGS_CREDENTIAL_BRIDGE_URL;
  const key = process.env.NEATLOGS_CREDENTIAL_BRIDGE_KEY;
  if (!endpoint || !key || options.token !== process.env.NEATLOGS_TOKEN || options.projectId !== process.env.NEATLOGS_PROJECT_ID ||
      new URL(options.baseUrl).origin !== process.env.NEATLOGS_HOST) return undefined;
  const url = new URL(endpoint);
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || url.pathname !== '/refresh' || url.search || url.hash || url.username || url.password) {
    throw new Error('Invalid CLI credential bridge.');
  }
  let token = options.token;
  let pending: Promise<string> | undefined;
  return async (rejected: string): Promise<string> => {
    if (token !== rejected) return token;
    pending ??= (async () => {
      try {
        const response = await context.with(suppressTracing(context.active()), () => fetch(url, { method: 'POST', redirect: 'error', signal: AbortSignal.timeout(5_000),
          headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ rejectedToken: rejected }) }));
        const reader = response.body?.getReader();
        if (!response.ok || !reader) { await response.body?.cancel(); throw new Error(); }
        const chunks: Uint8Array[] = [];
        let length = 0;
        try {
          for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            length += value.length;
            if (length > 1024) throw new Error();
            chunks.push(value);
          }
        } finally { await reader.cancel(); }
        token = z.object({ token: z.string().regex(/^nloa_[A-Za-z0-9_-]{43}$/) }).strict().parse(JSON.parse(Buffer.concat(chunks).toString('utf8'))).token;
        return token;
      } catch { throw new Error('CLI sign-in could not be renewed. Sign in again and resume the saved run.'); }
    })();
    try { return await pending; } finally { pending = undefined; }
  };
}
