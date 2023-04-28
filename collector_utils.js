function pad(num, ammount = 3) {
	let final = `${num}`;

	if (final.length >= ammount) {
		return final;
	}

	const needed = ammount - num.length;

	for (let i = 0; i < needed; i++) {
		final = '0' + final;
	}

	return final;
}

async function download_chapter(manga, chapter) {
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
download_chapter('ja_one_punch_man', '1');
