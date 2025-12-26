import io
from tqdm import tqdm
import concurrent.futures
import threading
from queue import Queue
from warcio.warcwriter import WARCWriter
from warcio.statusandheaders import StatusAndHeaders
import requests

INPUT_FILE = '/mnt/mnt/zjdx/tyhcyq/classifier/subsampled_positive_urls.txt'
OUTPUT_FILE = '/mnt/mnt/zjdx/tyhcyq/classifier/subsampled_positive_urls_1.warc.gz'
MAX_WORKERS = 128  # 可根据机器性能调整

def fetch_url(url):
	try:
		resp = requests.get(url, timeout=10)
		http_headers = [(k, v) for k, v in resp.headers.items()]
		warc_headers = StatusAndHeaders('200 OK', http_headers, protocol='HTTP/1.1')
		return (url, io.BytesIO(resp.content), warc_headers)
	except Exception as e:
		# print(f"Failed: {url} ({e})")
		return None

def writer_thread_func(queue, writer, done_event):
	while not done_event.is_set() or not queue.empty():
		try:
			item = queue.get(timeout=1)
		except Exception:
			continue
		if item is None:
			continue
		url, content, warc_headers = item
		record = writer.create_warc_record(
			url,
			'response',
			payload=content,
			http_headers=warc_headers
		)
		writer.write_record(record)
		# print(f"Saved: {url}")
		queue.task_done()

def main():
	queue = Queue(maxsize=MAX_WORKERS*2)
	done_event = threading.Event()
	with open(INPUT_FILE, 'r') as f, open(OUTPUT_FILE, 'wb') as out:
		writer = WARCWriter(out, gzip=True)
		writer_thread = threading.Thread(target=writer_thread_func, args=(queue, writer, done_event))
		writer_thread.start()
		urls = [line.strip() for line in f if line.strip()]
		# print(urls[0:10])
		# urls = urls[:1]
		with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
			futures = {executor.submit(fetch_url, url): url for url in urls}
			with tqdm(total=len(urls), desc='Fetching URLs') as pbar:
				for future in concurrent.futures.as_completed(futures):
					result = future.result()
					if result:
						queue.put(result)
					pbar.update(1)
		queue.join()
		done_event.set()
		writer_thread.join()

if __name__ == '__main__':
	main()
