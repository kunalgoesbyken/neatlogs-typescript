import { createHash } from 'node:crypto';
import { z } from 'zod';
import { DatasetApi } from '../dataset/api.js';

export const ALL_REPLAY_MAX_CASES = 2000;
export const selectedCaseSchema = z.object({ caseId: z.string().uuid(), revisionId: z.string().uuid(), position: z.number().int().nonnegative() }).strict();
export const datasetSelectionSchema = z.object({
  kind: z.literal('all'), groupId: z.string().uuid(), datasetId: z.string().uuid(), digest: z.string().regex(/^[a-f0-9]{64}$/),
  totalCases: z.number().int().min(1).max(ALL_REPLAY_MAX_CASES),
  batchOffset: z.number().int().min(0).max(ALL_REPLAY_MAX_CASES - 1), batchSize: z.number().int().min(1).max(100),
}).strict();
export type DatasetSelection = z.infer<typeof datasetSelectionSchema>;
export type SelectedCase = z.infer<typeof selectedCaseSchema>;
export function datasetSelectionDigest(input: { projectId: string; datasetId: string; versionId: string; cases: Array<Pick<SelectedCase, 'caseId' | 'revisionId'>> }) {
  return createHash('sha256').update(JSON.stringify(['neatlogs.local.all.v1', input.projectId, input.datasetId, input.versionId,
    input.cases.map(item => [item.caseId, item.revisionId])])).digest('hex');
}
const pageSchema = z.object({ cases: z.array(selectedCaseSchema.loose()).max(100), total: z.number().int().min(0).max(ALL_REPLAY_MAX_CASES) });
const overviewSchema = z.object({ dataset: z.object({ id: z.string().uuid(), caseCount: z.number().int().min(1).max(ALL_REPLAY_MAX_CASES) }), version: z.object({ id: z.string().uuid() }) });

/** Enumerate immutable IDs only; never retain dataset payloads between examples. */
export async function enumerateSavedDataset(api: DatasetApi, datasetId: string, versionId: string): Promise<SelectedCase[]> {
  const overview = overviewSchema.parse(await api.request('GET', `/datasets/${datasetId}?versionId=${versionId}`));
  if (overview.dataset.id !== datasetId || overview.version.id !== versionId) throw new Error('The dataset API returned a different frozen version.');
  const result: SelectedCase[] = [], identities = new Set<string>();
  while (result.length < overview.dataset.caseCount) {
    const offset = result.length;
    const page = pageSchema.parse(await api.request('GET', `/datasets/${datasetId}/cases?versionId=${versionId}&offset=${offset}&limit=100`));
    if (page.total !== overview.dataset.caseCount || page.cases.length !== Math.min(100, page.total - offset))
      throw new Error('The frozen dataset page is incomplete or its total changed.');
    for (const [index, item] of page.cases.entries()) {
      if (item.position !== offset + index || identities.has(item.caseId)) throw new Error('The frozen dataset contains repeated or out-of-order examples.');
      identities.add(item.caseId);
      result.push(selectedCaseSchema.parse({ caseId: item.caseId, revisionId: item.revisionId, position: item.position }));
    }
  }
  return result;
}
