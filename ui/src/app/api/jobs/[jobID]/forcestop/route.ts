import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
    const { jobID } = await params;

    const updatedJob = await prisma.job.update({
        where: { id: jobID },
        data: {
        status: 'stopped',
        stop: true,
        info: 'Job force stopped',
        },
    });

    return NextResponse.json(updatedJob);
}