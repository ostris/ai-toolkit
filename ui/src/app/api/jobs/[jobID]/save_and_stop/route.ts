import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

const handleSaveAndStop = async (jobID: string) => {
  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: "Job not found" }, { status: 404 });
  }

  if (job.status !== "running") {
    return NextResponse.json(
      { error: `Job is ${job.status}, must be running to save and stop` },
      { status: 400 }
    );
  }

  // Set both flags: save_now=1 and stop=1
  // The training loop's end_step_hook() will pick them up at the end of the current step:
  // 1. maybe_save() sees save_now=1, saves checkpoint, clears save_now
  // 2. maybe_stop() sees stop=1, marks as stopped, raises Exception to break loop
  await prisma.job.update({
    where: { id: jobID },
    data: { save_now: true, stop: true, info: "Saving checkpoint before stopping..." },
  });

  // Do NOT send SIGINT — let end_step_hook handle it naturally
  return NextResponse.json({
    jobID,
    message: "Checkpoint save and stop requested",
  });
};

export async function GET(request: NextRequest, ctx: { params: { jobID: string } }) {
  const { jobID } = await ctx.params;
  return handleSaveAndStop(jobID);
}

export async function POST(request: NextRequest, ctx: { params: { jobID: string } }) {
  const { jobID } = await ctx.params;
  return handleSaveAndStop(jobID);
}
