const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

async function migrateMultiGPU() {
  try {
    console.log('Starting multi-GPU migration...');

    // Get all existing jobs
    const jobs = await prisma.job.findMany();
    console.log(`Found ${jobs.length} jobs to migrate`);

    for (const job of jobs) {
      // Update job with default multi-GPU values
      await prisma.job.update({
        where: { id: job.id },
        data: {
          use_multi_gpu: false,
          accelerate_config: null,
          num_gpus: 1,
        },
      });
      console.log(`Migrated job: ${job.name}`);
    }

    console.log('Migration completed successfully!');
  } catch (error) {
    console.error('Migration failed:', error);
  } finally {
    await prisma.$disconnect();
  }
}

migrateMultiGPU(); 