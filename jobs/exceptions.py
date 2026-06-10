class JobControlFlowException(Exception):
    pass


class JobStoppedException(JobControlFlowException):
    pass


class JobReturnedToQueueException(JobControlFlowException):
    pass