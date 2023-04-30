function pad(num, ammount = 3) {
	let final = `${num}`;

	if (final.length >= ammount) {
		return final;
	}

	const needed = ammount - final.length;

	for (let i = 0; i < needed; i++) {
		final = '0' + final;
	}

	return final;
}

async function download_chapters_canvas(manga, chapter) {
	const download = async (el, filename, delay = 1000) => {
		el.scrollIntoView();
		await new Promise((res) => setTimeout(res, delay));
		const link = document.createElement('a');
		link.download = `${filename}.png`;
		link.href = el.toDataURL();
		link.click();
		link.remove();
		console.log('Downloading ', filename);
	};

	const items = document.querySelectorAll('canvas');
	for (let i = 0; i < items.length; i++) {
		await download(items[i], `${mana}_${chapter}_${pad(i)}`);
	}
}

async function toDataURL(url) {
	const blob = await fetch(url).then((res) => res.blob());
	return URL.createObjectURL(blob);
}

async function download_chapters_images(manga, chapter) {
	const download = async (el, filename, delay = 1000) => {
		el.scrollIntoView();
		await new Promise((res) => setTimeout(res, delay));
		const link = document.createElement('a');
		link.download = `${filename}.png`;
		link.href = await toDataURL(el.src);
		link.click();
		link.remove();
		console.log('Downloading ', filename);
	};

	const items = document.querySelectorAll('.image-vertical');
	for (let i = 0; i < items.length; i++) {
		await download(items[i], `${manga}_${chapter}_${pad(i)}`);
	}
}
