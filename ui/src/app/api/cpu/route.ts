import { NextResponse } from 'next/server';
import si from 'systeminformation';
import { CpuInfo } from '@/types';

export async function GET() {
  try {
    const cpuInfoRaw = await si.cpu();
    const memoryData = await si.mem();
    let cpuInfo: CpuInfo = {
      name: `${cpuInfoRaw.manufacturer} ${cpuInfoRaw.brand}`,
      cores: cpuInfoRaw.cores,
      temperature: (await si.cpuTemperature()).main || 0,
      totalMemory: memoryData.total / (1024 * 1024),
      availableMemory: memoryData.available / (1024 * 1024),
      freeMemory: memoryData.free / (1024 * 1024),
      currentLoad: (await si.currentLoad()).currentLoad || 0,
    };

    return NextResponse.json(cpuInfo);
  } catch (error) {
    console.error('Error fetching CPU stats:', error);
    return NextResponse.json(
      {
        error: `Failed to fetch CPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}
