import 'dart:math' as math;

class ChartViewport {
  final int totalCount;
  final int firstIndex;
  final int visibleCount;
  final int minVisibleCount;

  const ChartViewport({
    required this.totalCount,
    required this.firstIndex,
    required this.visibleCount,
    this.minVisibleCount = 24,
  });

  factory ChartViewport.live({
    required int totalCount,
    int preferredVisibleCount = 120,
    int minVisibleCount = 24,
  }) {
    final visible = _clampVisibleCount(
      totalCount: totalCount,
      visibleCount: preferredVisibleCount,
      minVisibleCount: minVisibleCount,
    );
    return ChartViewport(
      totalCount: totalCount,
      firstIndex: math.max(0, totalCount - visible),
      visibleCount: visible,
      minVisibleCount: minVisibleCount,
    );
  }

  int get visibleEndExclusive =>
      math.min(totalCount, firstIndex + visibleCount);

  bool get isAtLiveEnd => visibleEndExclusive >= totalCount;

  ChartViewport pan(int deltaBars) {
    final maxFirst = math.max(0, totalCount - visibleCount);
    return ChartViewport(
      totalCount: totalCount,
      firstIndex: (firstIndex + deltaBars).clamp(0, maxFirst),
      visibleCount: visibleCount,
      minVisibleCount: minVisibleCount,
    );
  }

  ChartViewport goLive() => ChartViewport(
        totalCount: totalCount,
        firstIndex: math.max(0, totalCount - visibleCount),
        visibleCount: visibleCount,
        minVisibleCount: minVisibleCount,
      );

  ChartViewport zoom(double factor, {double anchorFraction = 0.5}) {
    if (!factor.isFinite || factor <= 0 || totalCount <= 0) return this;
    final anchor = anchorFraction.clamp(0.0, 1.0);
    final anchorIndex = firstIndex + visibleCount * anchor;
    final nextVisible = _clampVisibleCount(
      totalCount: totalCount,
      visibleCount: (visibleCount / factor).round(),
      minVisibleCount: minVisibleCount,
    );
    final maxFirst = math.max(0, totalCount - nextVisible);
    final nextFirst = (anchorIndex - nextVisible * anchor).round();
    return ChartViewport(
      totalCount: totalCount,
      firstIndex: nextFirst.clamp(0, maxFirst),
      visibleCount: nextVisible,
      minVisibleCount: minVisibleCount,
    );
  }

  ChartViewport withTotalCount(int nextTotalCount) {
    final visible = _clampVisibleCount(
      totalCount: nextTotalCount,
      visibleCount: visibleCount,
      minVisibleCount: minVisibleCount,
    );
    final maxFirst = math.max(0, nextTotalCount - visible);
    final keepLive = isAtLiveEnd;
    return ChartViewport(
      totalCount: nextTotalCount,
      firstIndex: keepLive ? maxFirst : firstIndex.clamp(0, maxFirst),
      visibleCount: visible,
      minVisibleCount: minVisibleCount,
    );
  }
}

int _clampVisibleCount({
  required int totalCount,
  required int visibleCount,
  required int minVisibleCount,
}) {
  if (totalCount <= 0) return 0;
  final minCount = minVisibleCount.clamp(1, totalCount);
  return visibleCount.clamp(minCount, totalCount);
}
