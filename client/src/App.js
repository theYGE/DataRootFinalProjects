import React from 'react'
import axios, { post } from 'axios';
import Button from 'react-bootstrap/Button';

class SimpleReactFileUpload extends React.Component {

    constructor(props) {
        super(props);
        this.state ={
            file:null,
            message: '',
            shouldShowFirst: false,
            shouldShowSecond: false
        }
        this.onFormSubmit = this.onFormSubmit.bind(this)
        this.onChange = this.onChange.bind(this)
        this.fileUpload = this.fileUpload.bind(this)
        this.firstProjectOnClick = this.firstProjectOnClick.bind(this)
        this.secondProjectOnClick = this.secondProjectOnClick.bind(this)
        this.onTextareaChange = this.onTextareaChange.bind(this)
        this.reviewUpload = this.reviewUpload.bind(this)
        this.onSecondFormSubmit = this.onSecondFormSubmit.bind(this)
    }
    onFormSubmit(e){
        console.log('Hi');
        e.preventDefault() // Stop form submit
        this.fileUpload(this.state.file).then((response)=>{
            // this.setState({
            //     message: 'Changed'
            // })
            console.log(response.data);
            // this.setState({
            //     message: response.data
            // })
            this.setState({
                message: response.data.label
            })
        })
    }
    onChange(e) {
        this.setState({file:e.target.files[0]})
    }

    onSecondFormSubmit(e){

        e.preventDefault() // Stop form submit
        this.reviewUpload(this.state.review).then((response)=>{
            // this.setState({
            //     message: 'Changed'
            // })
            console.log(response.data);
            // this.setState({
            //     message: response.data
            // })
            this.setState({
                prob: response.data.prob
            })
        })
    }

    reviewUpload(review){
        const url = 'http://3.121.177.114:5000/second/predict';
        const formData = new FormData();
        formData.append('review',review)
        // const config = {
        //     headers: {
        //         'content-type': 'multipart/form-data'
        //     }
        // }
        return  post(url, formData)
    }

    onTextareaChange(e) {
        // this.setState({file:e.target.files[0]})
        console.log(e.target.value)
        this.setState({
            review: e.target.value
        })
    }

    firstProjectOnClick() {
        this.setState({
            shouldShowFirst: true,
            shouldShowSecond: false,
            review: "",
            prob: null
        })
    }

    secondProjectOnClick() {
        this.setState({
            shouldShowFirst: false,
            shouldShowSecond: true,
            message: ''
        })
    }

    fileUpload(file){
        const url = 'http://3.121.177.114:5000/first/predict';
        const formData = new FormData();
        formData.append('image',file)
        const config = {
            headers: {
                'content-type': 'multipart/form-data'
            }
        }
        return  post(url, formData,config)
    }

    render() {
        return (

            <div align="center">
                <h1>DataRoot University Final Projects</h1>
                <br/>
                <br/>
                <Button variant="primary" onClick={this.firstProjectOnClick}>First project</Button>
                <br/>
                <br/>
                <br/>
                <Button variant="primary" onClick={this.secondProjectOnClick}>Second project</Button>
                <br/>
                <br/>
                {this.state.shouldShowFirst
                    ?
                    <div>
                        <form onSubmit={this.onFormSubmit}>
                            <h1>First Project: Sign Classification</h1>
                            <h4>This project is dedicated to classifying hand signs from 0 to 5</h4>
                            <h4>Take a photo of your palm showing sign from 0 to 5, upload the image and see the result of classification</h4>
                            <br/>
                            <h5>Note: background of your photo should be plain for the best performance of the model</h5>
                            <br/>
                            <input type="file" onChange={this.onChange} />
                            <button type="submit">Upload</button>
                        </form>
                        <br/>
                        <h5>You showed sign: {this.state.message}</h5>
                    </div>
                    :
                    null
                }

                {this.state.shouldShowSecond
                    ?
                    <div>
                        <form onSubmit={this.onSecondFormSubmit}>
                            <h1>Second Project: Sentiment Analysis</h1>
                            <h4>This project is dedicated to classifying movie reviews</h4>
                            <h4>Write a review to a movie you recently watched and hit classify to see the result</h4>
                            <br/>
                            <h5>Note: you should use english</h5>
                            <br/>
                            <textarea rows={5} cols={65} onChange={this.onTextareaChange}></textarea>
                            <br/>
                            <button type="submit">Classify</button>
                        </form>
                        <br/>
                        <p>Note: reviews that have similar good/bad probabilities might be neutral</p>
                        <h5>Your review is good with probability: {this.state.prob}</h5>
                        <h5>Your review is bad with probability: {1 - parseFloat(this.state.prob)}</h5>
                    </div>
                    :
                    null
                }

            </div>
        )
    }
}



export default SimpleReactFileUpload
