export function instrumentSdkStreamIterator(
  stream: any,
  wrapIterator: (iterator: AsyncIterator<any>) => AsyncIterator<any>,
): boolean {
  if (typeof stream?.iterator !== 'function') return false;
  const sdkIterator = stream.iterator;
  stream.iterator = () => wrapIterator(sdkIterator.call(stream));
  return true;
}
