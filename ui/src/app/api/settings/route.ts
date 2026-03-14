import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { defaultTrainFolder, defaultDatasetsFolder } from '@/paths';
import { flushCache } from '@/server/settings';

const prisma = new PrismaClient();

export async function GET() {
  try {
    const settings = await prisma.settings.findMany();
    const settingsObject = settings.reduce((acc: any, setting) => {
      acc[setting.key] = setting.value;
      return acc;
    }, {});
    // if TRAINING_FOLDER is not set, use default
    if (!settingsObject.TRAINING_FOLDER || settingsObject.TRAINING_FOLDER === '') {
      settingsObject.TRAINING_FOLDER = defaultTrainFolder;
    }
    // if DATASETS_FOLDER is not set, use default
    if (!settingsObject.DATASETS_FOLDER || settingsObject.DATASETS_FOLDER === '') {
      settingsObject.DATASETS_FOLDER = defaultDatasetsFolder;
    }
    return NextResponse.json(settingsObject);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { HF_TOKEN, TRAINING_FOLDER, DATASETS_FOLDER, COMFYUI_URL, COMFYUI_INPUT_DIR, COMFYUI_OUTPUT_DIR } = body;

    // Upsert all settings
    await Promise.all([
      prisma.settings.upsert({
        where: { key: 'HF_TOKEN' },
        update: { value: HF_TOKEN },
        create: { key: 'HF_TOKEN', value: HF_TOKEN },
      }),
      prisma.settings.upsert({
        where: { key: 'TRAINING_FOLDER' },
        update: { value: TRAINING_FOLDER },
        create: { key: 'TRAINING_FOLDER', value: TRAINING_FOLDER },
      }),
      prisma.settings.upsert({
        where: { key: 'DATASETS_FOLDER' },
        update: { value: DATASETS_FOLDER },
        create: { key: 'DATASETS_FOLDER', value: DATASETS_FOLDER },
      }),
      prisma.settings.upsert({
        where: { key: 'COMFYUI_URL' },
        update: { value: COMFYUI_URL || '' },
        create: { key: 'COMFYUI_URL', value: COMFYUI_URL || '' },
      }),
      prisma.settings.upsert({
        where: { key: 'COMFYUI_INPUT_DIR' },
        update: { value: COMFYUI_INPUT_DIR || '' },
        create: { key: 'COMFYUI_INPUT_DIR', value: COMFYUI_INPUT_DIR || '' },
      }),
      prisma.settings.upsert({
        where: { key: 'COMFYUI_OUTPUT_DIR' },
        update: { value: COMFYUI_OUTPUT_DIR || '' },
        create: { key: 'COMFYUI_OUTPUT_DIR', value: COMFYUI_OUTPUT_DIR || '' },
      }),
    ]);

    flushCache();

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
