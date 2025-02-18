import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');

  try {
    if (id) {
      const training = await prisma.training.findUnique({
        where: { id },
      });
      return NextResponse.json(training);
    }

    const trainings = await prisma.training.findMany({
      orderBy: { created_at: 'desc' },
    });
    return NextResponse.json(trainings);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id, name, run_data } = body;

    if (id) {
      // Update existing training
      const training = await prisma.training.update({
        where: { id },
        data: {
          name,
          run_data: JSON.stringify(run_data),
        },
      });
      return NextResponse.json(training);
    } else {
      // Create new training
      const training = await prisma.training.create({
        data: {
          name,
          run_data: JSON.stringify(run_data),
        },
      });
      return NextResponse.json(training);
    }
  } catch (error) {
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
