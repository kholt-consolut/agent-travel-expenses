import {
  Router,
  type Request,
  type Response,
  type NextFunction,
  type Router as RouterType,
} from 'express';
import multer from 'multer';
import { authMiddleware, requireAuth } from '../middleware/auth';
import { ChatSDKError } from '@chat-template/core/errors';

export const filesRouter: RouterType = Router();

filesRouter.use(authMiddleware);

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // matches the express.json() limit in index.ts
const ALLOWED_MIME_TYPES = new Set(['application/pdf']);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_FILE_SIZE_BYTES },
});

/**
 * POST /api/files/upload - Accept a file attachment for the chat.
 *
 * Returns the file as a data URI rather than storing it anywhere server-side:
 * the Python agent backend (agent_server/attachment_processing.py) is what
 * extracts the PDF text and stages the file to SAP, once the chat message
 * carrying this attachment reaches it.
 */
filesRouter.post(
  '/upload',
  requireAuth,
  upload.single('file'),
  (req: Request, res: Response) => {
    const file = req.file;
    if (!file) {
      const response = new ChatSDKError('bad_request:api').toResponse();
      return res.status(response.status).json(response.json);
    }

    if (!ALLOWED_MIME_TYPES.has(file.mimetype)) {
      return res
        .status(400)
        .json({ error: `Unsupported file type: ${file.mimetype}` });
    }

    const dataUri = `data:${file.mimetype};base64,${file.buffer.toString('base64')}`;

    return res.status(200).json({
      url: dataUri,
      pathname: file.originalname,
      contentType: file.mimetype,
    });
  },
);

// Multer errors (e.g. file too large) land here instead of the generic error
// handler in index.ts, so the client gets the actual reason.
filesRouter.use(
  (err: unknown, _req: Request, res: Response, next: NextFunction) => {
    if (err instanceof multer.MulterError) {
      return res.status(400).json({ error: err.message });
    }
    next(err);
  },
);
