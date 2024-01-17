// script.js

function validateInput(event) {
    var inputValue = event.target.value;
    var arabicRegex = /[\u0600-\u06FF]/;

    if (!arabicRegex.test(inputValue)) {
        event.target.value = inputValue.replace(/[^؀-ۿ]/g, '');
    }
}
