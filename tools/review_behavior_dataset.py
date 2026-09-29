"""Local crop reviewer. Run with Python; no additional dependencies required."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import secrets
import shutil
import errno
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

CLASSES = {'sleep': '0', 'raise_hand': '1'}
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'output/6a1/left'


def atomic_write(path, data):
    temp = path.with_name(path.name + '.review-' + uuid4().hex)
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.csv = self.root / 'predictions.csv'
        self.lock = threading.RLock()
        self.history = self.root / '.review_history'
        self.read()

    def read(self):
        raw = self.csv.read_bytes()
        self.revision = hashlib.sha256(raw).hexdigest()
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        self.fields = reader.fieldnames
        self.rows = list(reader)
        if not self.fields or not {'image', 'crop', 'label_file', 'yolo_line', 'class_name', 'class_id'} <= set(self.fields):
            raise ValueError('predictions.csv thiếu các cột liên kết ảnh và nhãn')

    def path(self, row, key):
        folder = {'image': 'dataset_yolo/images', 'label_file': 'dataset_yolo/labels',
                  'crop': 'dataset_cls/' + row['class_name']}[key]
        if row['class_name'] not in CLASSES:
            raise ValueError('Lớp không hợp lệ')
        allowed_root = (self.root / folder).resolve()
        recorded = Path(row[key]).resolve()
        if recorded.is_file() and recorded.is_relative_to(allowed_root):
            return recorded
        # Basename fallback supports moved datasets and train/val/test folders.
        matches = list(allowed_root.rglob(Path(row[key]).name))
        if len(matches) != 1:
            raise ValueError(f'Không tìm thấy duy nhất file: {Path(row[key]).name}')
        path = matches[0].resolve()
        if not path.is_relative_to(allowed_root):
            raise ValueError('Đường dẫn không hợp lệ')
        return path

    def listing(self):
        with self.lock:
            self.read()
            counts = {name: sum(r['class_name'] == name for r in self.rows) for name in CLASSES}
            return dict(revision=self.revision, counts=counts, undo=self.latest() is not None,
                        items=[dict(id=i, name=Path(r['crop']).name, frame=Path(r['image']).name,
                                    class_name=r['class_name'], confidence=r.get('confidence', ''),
                                    box=list(map(float, r['yolo_line'].split()[1:])))
                               for i, r in enumerate(self.rows)])

    def latest(self):
        entries = sorted(self.history.glob('*/commit.json')) if self.history.exists() else []
        return entries[-1] if entries else None

    def check(self, revision):
        self.read()
        if revision != self.revision:
            raise ValueError('Dữ liệu đã thay đổi. Tải lại trang rồi thử lại.')

    def edit(self, index, action, revision):
        with self.lock:
            self.check(revision)
            if action not in (*CLASSES, 'delete'):
                raise ValueError('Thao tác không hợp lệ')
            if not 0 <= index < len(self.rows):
                raise ValueError('Crop không tồn tại')
            row = self.rows[index]
            if action == row['class_name']:
                return
            label, crop = self.path(row, 'label_file'), self.path(row, 'crop')
            old_label = label.read_bytes()
            lines = old_label.decode('utf-8').splitlines()
            matches = [i for i, line in enumerate(lines) if line.split() == row['yolo_line'].split()]
            if not matches:
                raise ValueError('Không tìm thấy box tương ứng; chưa sửa dữ liệu.')
            line_index = matches[0]
            destination = None
            if action == 'delete':
                del lines[line_index]
                del self.rows[index]
            else:
                stem = crop.stem
                suffix = '_' + row['class_name']
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)] + '_' + action
                destination = self.root / 'dataset_cls' / action / (stem + crop.suffix)
                if destination.exists():
                    raise ValueError('Tên crop đích đã tồn tại; chưa sửa dữ liệu.')
                fields = lines[line_index].split()
                fields[0] = CLASSES[action]
                lines[line_index] = ' '.join(fields)
                row.update(class_id=CLASSES[action], class_name=action,
                           crop=str(destination), yolo_line=lines[line_index])
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)
            csv_data = stream.getvalue().encode('utf-8-sig')
            # Store originals before modifying anything; frames are never written.
            import time
            backup = self.history / (str(time.time_ns()) + '-' + uuid4().hex)
            backup.mkdir(parents=True)
            shutil.copy2(self.csv, backup / 'predictions.csv')
            shutil.copy2(label, backup / 'label.txt')
            shutil.copy2(crop, backup / 'crop')
            metadata = dict(label=str(label), crop=str(crop), destination=str(destination) if destination else None,
                            after=hashlib.sha256(csv_data).hexdigest())
            new_label = ('\n'.join(lines) + ('\n' if lines else '')).encode('utf-8')
            metadata['label_after'] = hashlib.sha256(new_label).hexdigest()
            metadata['crop_hash'] = hashlib.sha256((backup / 'crop').read_bytes()).hexdigest()
            (backup / 'prepared.json').write_text(json.dumps(metadata))
            try:
                if destination:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    crop.rename(destination)
                else:
                    crop.unlink()
                atomic_write(label, new_label)
                atomic_write(self.csv, csv_data)
                (backup / 'prepared.json').rename(backup / 'commit.json')
            except Exception:
                atomic_write(label, old_label)
                atomic_write(self.csv, (backup / 'predictions.csv').read_bytes())
                shutil.copy2(backup / 'crop', crop)
                if destination:
                    destination.unlink(missing_ok=True)
                raise

    def undo(self, revision):
        with self.lock:
            self.check(revision)
            commit = self.latest()
            if commit is None:
                raise ValueError('Không có thao tác để hoàn tác')
            meta = json.loads(commit.read_text())
            if self.revision != meta['after']:
                raise ValueError('CSV đã được sửa bên ngoài; không thể hoàn tác tự động')
            label, crop = Path(meta['label']), Path(meta['crop'])
            destination = Path(meta['destination']) if meta['destination'] else None
            if hashlib.sha256(label.read_bytes()).hexdigest() != meta['label_after'] or crop.exists():
                raise ValueError('File đã thay đổi bên ngoài; không thể hoàn tác tự động')
            if destination and hashlib.sha256(destination.read_bytes()).hexdigest() != meta['crop_hash']:
                raise ValueError('Crop đã thay đổi bên ngoài')
            backup = commit.parent
            shutil.copy2(backup / 'crop', crop)
            atomic_write(label, (backup / 'label.txt').read_bytes())
            atomic_write(self.csv, (backup / 'predictions.csv').read_bytes())
            if destination:
                destination.unlink()
            commit.rename(backup / 'undone.json')


def handler_for(dataset):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, kind='application/json'):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == '/':
                    html = Path(__file__).with_suffix('.html').read_text().replace('__TOKEN__', token)
                    self.send(200, html.encode(), 'text/html; charset=utf-8')
                elif url.path == '/api/items':
                    self.send(200, dataset.listing())
                elif url.path == '/api/image':
                    query = parse_qs(url.query)
                    with dataset.lock:
                        dataset.check(query['revision'][0])
                        index = int(query['id'][0])
                        if not 0 <= index < len(dataset.rows):
                            raise ValueError('Crop không tồn tại')
                        kind = query['kind'][0]
                        if kind not in ('image', 'crop'):
                            raise ValueError('Loại ảnh không hợp lệ')
                        path = dataset.path(dataset.rows[index], kind)
                        self.send(200, path.read_bytes(), mimetypes.guess_type(path)[0] or 'image/jpeg')
                else:
                    self.send(404, {'error': 'Not found'})
            except (ValueError, OSError, KeyError, IndexError) as exc:
                self.send(409, {'error': str(exc)})

        def do_POST(self):
            if self.headers.get('X-Review-Token') != token:
                self.send(403, {'error': 'Invalid token'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 4096:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if self.path == '/api/edit':
                    dataset.edit(int(body['id']), body['action'], body['revision'])
                elif self.path == '/api/undo':
                    dataset.undo(body['revision'])
                else:
                    self.send(404, {'error': 'Not found'})
                    return
                self.send(200, {'ok': True})
            except (ValueError, OSError, KeyError, IndexError) as exc:
                self.send(409, {'error': str(exc)})

    return Handler


def create_server(host, port, handler, attempts=20):
    """Bind the requested port, falling forward when another reviewer uses it."""
    if port == 0:
        return ThreadingHTTPServer((host, 0), handler)
    for candidate in range(port, port + attempts):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise OSError(
        errno.EADDRINUSE,
        f'Các cổng {port}-{port + attempts - 1} đều đang được sử dụng',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    dataset = Dataset(args.root)
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    server = create_server('127.0.0.1', args.port, handler_for(dataset))
    actual_port = server.server_address[1]
    if actual_port != args.port and args.port != 0:
        print(f'Cổng {args.port} đang bận; chuyển sang cổng {actual_port}.', flush=True)
    print(f'Review {len(dataset.rows)} crops: http://127.0.0.1:{actual_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
