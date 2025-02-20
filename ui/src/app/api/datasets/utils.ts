import { PrismaClient } from '@prisma/client';
import { defaultDatasetsFolder } from '@/paths';

const prisma = new PrismaClient();

export const getDatasetsRoot = async () => {
  let row = await prisma.settings.findFirst({
    where: {
      key: 'DATASETS_FOLDER',
    },
  });
  let datasetsPath = defaultDatasetsFolder;
  if (row?.value && row.value !== '') {
    datasetsPath = row.value;
  }
  return datasetsPath;
};